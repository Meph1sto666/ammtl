import random # type: ignore
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from lib.types.bubble import Bubble
from datetime import datetime as dt
from lib.misc import CropBox, drawBoundingBox
import cv2.typing
import manga_ocr # type: ignore
from PIL import Image # type: ignore

class Page:
	def __init__(self, img:cv2.Mat, mocr:manga_ocr.MangaOcr) -> None:
		self.mocr:manga_ocr.MangaOcr = mocr
		self.img:cv2.Mat = img
		self.area:int = img.shape[0]*img.shape[1]
		start:dt = dt.now()
		self.mask:cv2.Mat = self.createMask(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)) # type: ignore
		self.bubbles:list[Bubble] = self.conjectBubbles()
		self.bubbles = [b for b in self.bubbles if b.hasContent]
		# for b in self.bubbles:
		# 	b.translate()
		end:dt = dt.now()
		self.img = self.drawTextBounds(self.img)
		self.tDelta:float = (end-start).total_seconds()*1000

	def createMask(self, img:cv2.Mat) -> cv2.Mat:
		threshed:cv2.Mat = cv2.threshold(cv2.medianBlur(img, 5), 200, 255, cv2.THRESH_BINARY)[1] # type: ignore // reduced threshold less black
		morphed:cv2.Mat = cv2.morphologyEx(threshed, cv2.MORPH_RECT, cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))) # type: ignore // (9,9) < -> less thick > -> thicker
		contours:list[cv2.Mat] = cv2.findContours(threshed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0] # type: ignore
		filteredContours:list[cv2.Mat] = [contour for contour in contours if cv2.contourArea(contour) > 15000 and cv2.mean(morphed, mask=cv2.bitwise_not(morphed))[0] == 0]
		mask:cv2.Mat = np.zeros_like(morphed)
		cv2.drawContours(mask, filteredContours, -1, 1, thickness=cv2.FILLED) # type: ignore
		morphed = cv2.bitwise_not(morphed) # type: ignore
		return cv2.bitwise_and(morphed, morphed, mask=mask) # type: ignore

	def conjectBubbles(self) -> list[Bubble]:
		bbs:list[Bubble] = []
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		gray = cv2.bitwise_and(gray, self.mask)
		
		blurred:cv2.Mat = cv2.medianBlur(gray, 3) # type: ignore
		threshed:cv2.Mat = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1] # type: ignore

		contours:list[cv2.Mat] = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] # type: ignore
		self.img=cv2.cvtColor(threshed, cv2.COLOR_GRAY2BGR) # type: ignore		
		filtered:list[cv2.Mat] = []
		for c in range(len(contours)):
			if len(contours[c]) <= 0: continue
			perimeter = cv2.arcLength(contours[c], True)
			# color:tuple[int,int,int] = (0,255,255) if perimeter < 550 else (255,0,255)
			# color:tuple[int,int,int] = (random.randint(1,256),random.randint(1,256),random.randint(1,256))
			if perimeter > self.img.shape[1]*.42: continue
			area:float = cv2.contourArea(contours[c])
			if not self.area*.00003 < area < self.area*.005: continue
			# approx = cv2.approxPolyDP(contours[c], 0.01 * cv2.arcLength(contours[c], True), True)

			# color:tuple[int,int,int] = (0,255,255) if len(approx) < 20 else (255,0,255)

			# if len(approx) !=12: continue
			# if (abs(len(contours[c])-len(contours[c-1])) > 5 and abs(len(contours[c])-len(contours[c+1])) > 5) if c+1<len(contours) else False: continue 
			# cv2.drawContours(self.img, [contours[c]], -1, color, 2)
			filtered.append(contours[c])
		a = 255/ len(filtered)
		for c in range(len(filtered)):
			color:tuple[int,int,int] = (0,255,a*c) if c%2==0 else (255,0,a*c)
			drawBoundingBox(self.img, *cv2.boundingRect(filtered[c]), text=str(c), color=color)
		self.cluster(filtered)
		return bbs

	def cluster(self, contours:list[cv2.Mat]) -> list[cv2.Mat]:
		clusters:dict[int, list[tuple[float, int, int]]] = {}
		c:int = 0
		for i in range(len(contours)-1):
			M1 = cv2.moments(contours[i])
			cX1 = int(M1['m10'] / M1['m00'])
			cY1 = int(M1['m01'] / M1['m00'])
			M2 = cv2.moments(contours[i+1])
			cX2 = int(M2['m10'] / M2['m00'])
			cY2 = int(M2['m01'] / M2['m00'])
			distance = ((cX2 - cX1)**2 + (cY2 - cY1)**2)**0.5
			if distance > 100: c+=1
			try: clusters[c].append((distance, i, i+1))
			except: clusters[c] = [(distance, i, i+1)]
		for sc in clusters.items():
			print(sc)


	def conjectBubblesByBlobs(self) -> list[Bubble]:
		bbs:list[Bubble] = []
		# mask:cv2.Mat = np.zeros((self.img.shape[0],self.img.shape[1]), dtype=np.uint8)
		gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		gray = cv2.bitwise_and(gray, self.mask)
		blurred:cv2.Mat = cv2.GaussianBlur(gray, (3,3), 2) # type: ignore // sigma 0, 1 or 2
		threshed:cv2.Mat = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1] # type: ignore
		blurred2:cv2.Mat = cv2.GaussianBlur(threshed, (3,3), 2) # type: ignore // sigma 0, 1 or 2
		self.img=blurred2
		keypoints = cv2.AgastFeatureDetector_create( # type: ignore
			threshold=64,
			nonmaxSuppression=False,
			type=cv2.AGAST_FEATURE_DETECTOR_AGAST_5_8 # type: ignore
		).detect(blurred2)
		labels:cv2.Mat = DBSCAN( # type: ignore
			eps=30,
			min_samples=16
		).fit_predict(np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints])) # type: ignore
		"""
					threshold=120,
			nonmaxSuppression=True,
			type=cv2.AGAST_FEATURE_DETECTOR_OAST_9_16
		).detect(threshed)
		labels:cv2.Mat = DBSCAN( # type: ignore
			eps=35,
			min_samples=8
		
		"""
		"""
				keypoints = cv2.AgastFeatureDetector_create( # type: ignore
			threshold=150,
			nonmaxSuppression=False,
			type=cv2.AGAST_FEATURE_DETECTOR_OAST_9_16 # type: ignore
		).detect(threshed)
		labels:cv2.Mat = DBSCAN( # type: ignore
			eps=35,
			min_samples=30
		).fit_predict(np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints])) # type: ignore
		"""
		for label in np.unique(labels): # type: ignore
			if label == -1: continue
			clusterPoints:cv2.Mat = np.array([(kp.pt[0], kp.pt[1]) for i, kp in enumerate(keypoints) if labels[i] == label]).astype(np.int32) # type: ignore
			if len(clusterPoints) < 3: continue
			br:tuple[int,...] = cv2.boundingRect(clusterPoints) # type: ignore
			if not 0.075*self.area > br[2]*br[3] > 0.0005*self.area: continue
			bbs.append(Bubble(self.img, CropBox(*br, tolerance=20), self.mocr))

			# cv2.rectangle(mask, (br[0]-20, br[1]-20), (br[0] + br[2]+40, br[1] + br[3]+40), (255), -1) # type: ignore
		self.img = cv2.drawKeypoints(self.img, keypoints, None, (255,0,255)) # type: ignore
		return bbs

	def drawTextBounds(self, img:cv2.Mat) -> cv2.Mat:
		for b in self.bubbles:
			# print(b)
			img = drawBoundingBox(img, *b.box.toTuple(), text=str(round(b.area/(len(b.keyPoints)+1),1)), inset=False)
			# img = drawBoundingBox(img, *b.box.toTuple())
		return img