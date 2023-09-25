import random
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from lib.types.bubble import Bubble
# from datetime import datetime as dt
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
		# self.out = self.drawTextBounds(self.out)

	def createMask(self) -> cv2.typing.MatLike:
		mask = cv2.bitwise_and(self.__gray__,self.__gray__,mask=cv2.bitwise_not(self.__gray__))
		mask:cv2.typing.MatLike = cv2.stackBlur(mask, (9,9))
		mask:cv2.typing.MatLike = cv2.medianBlur(mask, 19)
		mask = cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_DILATE,(3,3)))
		mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
		return cv2.bitwise_not(mask)

	def conjectBubbles(self) -> list[Bubble]:
		bbs:list[Bubble] = []
		gray:cv2.typing.MatLike = self.__gray__
		gray = cv2.bitwise_and(gray, self.mask)
		# blurred:cv2.typing.MatLike = cv2.medianBlur(gray, self.preset.conjectionBlur)
		# threshed:cv2.typing.MatLike = cv2.threshold(blurred, self.preset.conjectionThresh, 255, cv2.THRESH_BINARY)[1]
		contours:list[cv2.typing.MatLike] = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] # type: ignore
		filtered:list[cv2.typing.MatLike] = []
		# filtered = contours
		for c in range(len(contours)):
			if len(contours[c]) <= self.preset.conjectionMinContourLen: continue
			perimeter:float = cv2.arcLength(contours[c], True)
			if perimeter > self.img.shape[1]*self.preset.conjectionMaxContourPermitter: continue
			area:float = cv2.contourArea(contours[c])
			if not self.area*0 < area < self.area*.01: continue
			# if not self.area*self.preset.conjectionMinContourArea < area < self.area*self.preset.conjectionMaxContourArea: continue
			filtered.append(contours[c])
		cv2.drawContours(self.out, filtered, -1, color=(255,0,255), thickness=2)
		
		"""
		if len(filtered) < 1: return bbs
		contourFeatures:list[list[int]] = []
		for c in range(len(filtered)):
			moments:cv2.typing.Moments = cv2.moments(filtered[c]) # type: ignore
			cX:int = int(moments['m10'] / (moments['m00'] if moments['m00']>0 else 1))
			cY:int = int(moments['m01'] / (moments['m00'] if moments['m00']>0 else 1))
			contourFeatures.append([c,cX,cY])
		scan = DBSCAN(eps=self.preset.conjectionClusterEps*self.width, min_samples=self.preset.conjectionClusterSamples, algorithm=self.preset.conjectionClusterAlgorithm)
		labels:cv2.typing.MatLike = scan.fit_predict(np.asarray(contourFeatures, dtype=np.int32)) # type: ignore
		# ls: int = len(list(filter(lambda x: x!=-1, labels)))
		for l in np.unique(labels): # type: ignore
			if l == -1: continue
			clusterI = np.where(labels==l)[0] # type: ignore
			ccf:list[int] = [contourFeatures[i][0] for i in clusterI]
			clustered:list[cv2.typing.MatLike] = [filtered[i] for i in ccf]
			# if len(ccf) > 400: continue
			color:tuple[int, int, int] = (random.randint(0,256),random.randint(0,256),random.randint(0,256))
			for c in clustered:
				cv2.drawContours(self.out, [c], -1, color=color, thickness=2)
			brs:list[cv2.typing.Rect] = [cv2.boundingRect(c) for c in clustered]
			crds:list[list[int]] = [[i[e]+i[e-2] if e > 1 else i[e] for i in brs] for e in range(len(brs[0]))]
			bubbleBounds:cv2.typing.Rect = [min(crds[0]), min(crds[1]), max(crds[2])-min(crds[0]), max(crds[3])-min(crds[1])]
			if not self.area*self.preset.conjectionClusterMinArea < bubbleBounds[2]*bubbleBounds[3] < self.area*self.preset.conjectionClusterMaxArea: continue
			bbs.append(Bubble(self.img, CropBox(*bubbleBounds,tolerance=self.preset.conjectionBubbleTolerance), self.mocr, preset=self.preset)) # type: ignore
		"""
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
def interpolate(x:float, x1:float, y1:float, x2:float, y2:float) -> float:
	return y1+(x-x1)*((y2-y1)/(x2-x1))