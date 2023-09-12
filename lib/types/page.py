import typing
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from lib.types.bubble import Bubble
from datetime import datetime as dt
from lib.misc import CropBox, drawBoundingBox
import cv2.typing
import manga_ocr # type: ignore
from PIL import Image # type: ignore

class Preset:
	def __init__(self) -> None:
		self.maskBlur:int = 5 # 3-13 // try
		self.maskThresh:int = 200 # 50-254 // try
		self.maskMorph:tuple[int,int] = (7,7) # 3-13 // try
		self.maskContourFilterMaxArea:float = .0058 # 0.002-0.01 // maybe try if not too much
		self.conjectionBlur:int = 3 # 3-13 // try
		self.conjectionThresh:int = 170 # 50-254 // try
		self.conjectionMinContourLen:int = 0 # including // maybe try if not too much
		self.conjectionMaxContourPermitter:float = 0.42 # uses image width // try
		self.conjectionMinContourArea:float = .00003 # uses image area
		self.conjectionMaxContourArea:float = .005 # uses image area
		self.conjectionClusterEps:float = 0.035 # uses img width // try
		self.conjectionClusterSamples:int = 2 # 1 or more // try a bit
		self.conjectionClusterAlgorithm:typing.Literal['auto', 'ball_tree', 'kd_tree', 'brute'] = "auto"
		self.conjectionClusterMinArea:float = 0.0003895
		self.conjectionClusterMaxArea:float = 0.0389553
		self.conjectionBubbleTolerance:int = 20 # only for the bubble borders

class Page:
	def __init__(self, img:cv2.typing.MatLike, mocr:manga_ocr.MangaOcr, preset:Preset) -> None:
		self.preset:Preset = preset
		self.mocr:manga_ocr.MangaOcr = mocr
		self.img:cv2.typing.MatLike = img
		self.area:int = img.shape[0]*img.shape[1]
		self.width:int = img.shape[1]
		start:dt = dt.now()
		self.mask:cv2.typing.MatLike = self.createMask(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)) # type: ignore
		self.bubbles:list[Bubble] = self.conjectBubbles()
		self.bubbles = [b for b in self.bubbles if b.hasContent]
		# for b in self.bubbles:
		# 	b.translate()
		end:dt = dt.now()
		self.img = self.drawTextBounds(self.img)
		self.tDelta:float = (end-start).total_seconds()*1000

	def createMask(self, img:cv2.typing.MatLike) -> cv2.typing.MatLike:
		threshed:cv2.typing.MatLike = cv2.threshold(cv2.medianBlur(img, self.preset.maskBlur), self.preset.maskThresh, 255, cv2.THRESH_BINARY)[1] # type: ignore // reduced threshold less black
		morphed:cv2.typing.MatLike = cv2.morphologyEx(threshed, cv2.MORPH_RECT, cv2.getStructuringElement(cv2.MORPH_RECT,self.preset.maskMorph)) # type: ignore // (9,9) < -> less thick > -> thicker
		contours:list[cv2.typing.MatLike] = cv2.findContours(threshed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0] # type: ignore
		filteredContours:list[cv2.typing.MatLike] = [contour for contour in contours if cv2.contourArea(contour) > self.preset.maskContourFilterMaxArea*self.area and cv2.mean(morphed, mask=cv2.bitwise_not(morphed))[0] == 0]
		mask:cv2.typing.MatLike = np.zeros_like(morphed)
		cv2.drawContours(mask, filteredContours, -1, 1, thickness=cv2.FILLED) # type: ignore
		morphed = cv2.bitwise_not(morphed) # type: ignore
		return cv2.bitwise_and(morphed, morphed, mask=mask) # type: ignore

	def conjectBubbles(self) -> list[Bubble]:
		bbs:list[Bubble] = []
		gray:cv2.typing.MatLike = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		gray = cv2.bitwise_and(gray, self.mask)
		blurred:cv2.typing.MatLike = cv2.medianBlur(gray, self.preset.conjectionBlur) # type: ignore
		threshed:cv2.typing.MatLike = cv2.threshold(blurred, self.preset.conjectionThresh, 255, cv2.THRESH_BINARY)[1] # type: ignore
		contours:list[cv2.typing.MatLike] = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] # type: ignore
		self.img=cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR) # type: ignore
		filtered:list[cv2.typing.MatLike] = []
		for c in range(len(contours)):
			if len(contours[c]) <= self.preset.conjectionMinContourLen: continue
			perimeter:float = cv2.arcLength(contours[c], True)
			if perimeter > self.img.shape[1]*self.preset.conjectionMaxContourPermitter: continue
			area:float = cv2.contourArea(contours[c])
			if not self.area*self.preset.conjectionMinContourArea < area < self.area*self.preset.conjectionMaxContourArea: continue
			filtered.append(contours[c])
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
			# for c in clustered:
			# 	cv2.drawContours(self.img, [c], -1, color=color, thickness=2)
			brs:list[cv2.typing.Rect] = [cv2.boundingRect(c) for c in clustered]
			crds:list[list[int]] = [[i[e]+i[e-2] if e > 1 else i[e] for i in brs] for e in range(len(brs[0]))]
			bubbleBounds:cv2.typing.Rect = [min(crds[0]), min(crds[1]), max(crds[2])-min(crds[0]), max(crds[3])-min(crds[1])]

			if not self.area*self.preset.conjectionClusterMinArea < bubbleBounds[2]*bubbleBounds[3] < self.area*self.preset.conjectionClusterMaxArea: continue
			bbs.append(Bubble(self.img, CropBox(*bubbleBounds,tolerance=self.preset.conjectionBubbleTolerance), self.mocr))
		return bbs

	def drawTextBounds(self, img:cv2.typing.MatLike) -> cv2.typing.MatLike:
		for b in self.bubbles:
			# print(b)
			img = drawBoundingBox(img, *b.box.toTuple(), text=str(round(b.area/(len(b.keyPoints)+1),1)), inset=False)
		return img