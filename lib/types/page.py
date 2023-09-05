import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from lib.types.bubble import Bubble
from datetime import datetime as dt
from lib.misc import CropBox, drawBoundingBox
import manga_ocr # type: ignore

class Page:
	def __init__(self, img:cv2.Mat, mocr:manga_ocr.MangaOcr) -> None:
		self.mocr:manga_ocr.MangaOcr = mocr
		self.img:cv2.Mat = img
		self.area:int = img.shape[0]*img.shape[1]
		self.bubbles:list[Bubble] = self.conjectBubbles()
		start:dt = dt.now()
		self.bubbles = [b for b in self.bubbles if b.hasContent]
		# for b in self.bubbles:
		# 	b.translate()
		end:dt = dt.now()
		self.img = self.drawTextBounds(self.img)
		# Image.fromarray(self.img).show()
		print(f"{(end-start).total_seconds()}s")

		
	def conjectBubbles(self) -> list[Bubble]:
		bbs:list[Bubble] = []
		# mask:cv2.Mat = np.zeros((self.img.shape[0],self.img.shape[1]), dtype=np.uint8)
		blurred:cv2.Mat = cv2.GaussianBlur(self.img, (3,3), 2) # type: ignore // sigma 0, 1 or 2
		threshed = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
		self.img=threshed
		keypoints = cv2.AgastFeatureDetector_create(
			threshold=150,
			nonmaxSuppression=True,
			type=cv2.AGAST_FEATURE_DETECTOR_OAST_9_16
		).detect(threshed)
		labels:cv2.Mat = DBSCAN( # type: ignore
			eps=35,
			min_samples=30
		).fit_predict(np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints]))
		"""
					threshold=120,
			nonmaxSuppression=True,
			type=cv2.AGAST_FEATURE_DETECTOR_OAST_9_16
		).detect(threshed)
		labels:cv2.Mat = DBSCAN( # type: ignore
			eps=35,
			min_samples=8
		"""
		for label in np.unique(labels): # type: ignore
			if label == -1: continue
			clusterPoints:cv2.Mat = np.array([(kp.pt[0], kp.pt[1]) for i, kp in enumerate(keypoints) if labels[i] == label]).astype(np.int32)
			if len(clusterPoints) < 3: continue
			br:tuple[int,...] = cv2.boundingRect(clusterPoints) # type: ignore
			# if not 0.075*imgArea > br[2]*br[3] > 0.0005*imgArea: continue
			bbs.append(Bubble(self.img, CropBox(*br, tolerance=20), self.mocr))

			# cv2.rectangle(mask, (br[0]-20, br[1]-20), (br[0] + br[2]+40, br[1] + br[3]+40), (255), -1) # type: ignore
		self.img = cv2.drawKeypoints(self.img, keypoints, None, (255,0,255))
		return bbs

	def drawTextBounds(self, img:cv2.Mat) -> cv2.Mat:
		for b in self.bubbles:
			# print(b)
			img = drawBoundingBox(img, *b.box.toTuple(), text=str(b.area/(len(b.keyPoints)+1)), inset=False)
			# img = drawBoundingBox(img, *b.box.toTuple())
		return img