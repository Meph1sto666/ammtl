import random
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from lib.types.bubble import Bubble
from lib.misc import CropBox, drawBoundingBox
import cv2.typing
# import manga_ocr # type: ignore
from PIL import Image # type: ignore
from lib.types.preset import Preset

class Page:
	def __init__(self, img:cv2.typing.MatLike, mocr, preset:Preset) -> None: # type: ignore
		self.preset:Preset = preset
		self.mocr = mocr # type: ignore
		self.img:cv2.typing.MatLike = img
		self.__gray__:cv2.typing.MatLike = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) if len(self.img.shape) > 2 else self.img
		self.out:cv2.typing.MatLike = img.copy()
		self.area:int = img.shape[0]*img.shape[1]
		self.width:int = img.shape[1]
		self.height:int = img.shape[0]
		self.mask:cv2.typing.MatLike = self.createMask()
		self.bubbles:list[Bubble] = self.conjectBubbles()
		# self.bubbles = [b for b in self.bubbles if b.hasContent]

		# for b in range(len(self.bubbles)):
		# 	Image.fromarray(self.bubbles[b].img).save(f"./out/b/{f}_{b}.jpg")
		# 	b.translate()
		self.out = self.drawTextBounds(self.out)

	def createMask(self) -> cv2.typing.MatLike:
		blurred:cv2.typing.MatLike = cv2.medianBlur(self.__gray__, self.preset.maskBlur)
		threshed:cv2.typing.MatLike = cv2.threshold(blurred, self.preset.maskThresh, 255, cv2.THRESH_BINARY)[1] # // reduced threshold less black
		morphed:cv2.typing.MatLike = cv2.morphologyEx(threshed, cv2.MORPH_RECT, cv2.getStructuringElement(cv2.MORPH_RECT,self.preset.maskMorph)) # // (9,9) < -> less thick > -> thicker
		contours:list[cv2.typing.MatLike] = cv2.findContours(morphed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0] # type: ignore
		meanS:float = cv2.mean(morphed, mask=cv2.bitwise_not(morphed))[0]
		filteredContours:list[cv2.typing.MatLike] = [contour
            for contour in contours
            if (
            	cv2.contourArea(contour) > self.preset.maskContourFilterMaxArea*self.area
            ) and meanS == 0
        ]
		mask:cv2.typing.MatLike = np.zeros_like(morphed)
		cv2.drawContours(mask, filteredContours, -1, 1, thickness=cv2.FILLED) # type: ignore
		morphed = cv2.bitwise_not(morphed) # type: ignore
		masked: cv2.typing.MatLike = cv2.bitwise_and(morphed, morphed, mask=mask)
		# blurred2 = cv2.erode(masked, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=1)
		return masked

	def conjectBubbles(self) -> list[Bubble]:
		bbs:list[Bubble] = []
		# gray:cv2.typing.MatLike = self.__gray__
		# gray = cv2.bitwise_and(gray, self.mask)
		# threshed:cv2.typing.MatLike = cv2.threshold(gray, self.preset.conjectionThresh, 255, cv2.THRESH_BINARY)[1]
		contours:list[cv2.typing.MatLike] = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] # type: ignore
		# filtered:list[cv2.typing.MatLike] = []
		# for c in range(len(contours)):
		# 	if len(contours[c]) <= self.preset.conjectionMinContourLen: continue
		# 	perimeter:float = cv2.arcLength(contours[c], True)
		# 	if perimeter > self.img.shape[1]*self.preset.conjectionMaxContourPermitter: continue
		# 	area:float = cv2.contourArea(contours[c])
		# 	if not self.area*self.preset.conjectionMinContourArea < area < self.area*self.preset.conjectionMaxContourArea: continue
		# 	filtered.append(contours[c])
		filtered = contours
		if len(filtered) < 1: return bbs
		contourFeatures:list[list[int]] = []
		for c in range(len(filtered)):
			moments:cv2.typing.Moments = cv2.moments(filtered[c]) # type: ignore
			cX:int = int(moments['m10'] / (moments['m00'] if moments['m00']>0 else 1))
			cY:int = int(moments['m01'] / (moments['m00'] if moments['m00']>0 else 1))
			contourFeatures.append([c,cX,cY])
		scan = DBSCAN(eps=self.preset.conjectionClusterEps*self.width, min_samples=self.preset.conjectionClusterSamples, algorithm=self.preset.conjectionClusterAlgorithm)
		labels:cv2.typing.MatLike = scan.fit_predict(np.asarray(contourFeatures, dtype=np.int32)) # type: ignore
		for l in np.unique(labels): # type: ignore
			if l == -1: continue
			clusterI = np.where(labels==l)[0] # type: ignore
			ccf:list[int] = [contourFeatures[i][0] for i in clusterI]
			clustered:list[cv2.typing.MatLike] = [filtered[i] for i in ccf]
			# color = (random.randint(0,256),random.randint(0,256),random.randint(0,256))
			# for c in clustered:
			# 	cv2.drawContours(self.out, [c], -1, color=color, thickness=2)
			brs:list[cv2.typing.Rect] = [cv2.boundingRect(c) for c in clustered]
			crds:list[list[int]] = [[i[e]+i[e-2] if e > 1 else i[e] for i in brs] for e in range(len(brs[0]))]
			bubbleBounds:cv2.typing.Rect = [min(crds[0]), min(crds[1]), max(crds[2])-min(crds[0]), max(crds[3])-min(crds[1])]
			if not self.area*self.preset.conjectionClusterMinArea < bubbleBounds[2]*bubbleBounds[3] < self.area*self.preset.conjectionClusterMaxArea: continue
			bbs.append(Bubble(self.img, CropBox(*bubbleBounds,tolerance=self.preset.conjectionBubbleTolerance), self.mocr, preset=self.preset)) # type: ignore
		return bbs

	def update(self, preset:Preset|None=None) -> None:
		if preset != None:
			self.preset = preset
		self.mask = self.createMask()
		self.bubbles = self.conjectBubbles()

	def drawTextBounds(self, img:cv2.typing.MatLike) -> cv2.typing.MatLike:
		for b in self.bubbles:
			img = drawBoundingBox(img, *b.box.toTuple(), text=str(b.area), color=(random.randint(1,256),random.randint(1,256),random.randint(1,256)))
		return img