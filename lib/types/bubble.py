import cv2
import cv2.typing
import deep_translator
import numpy as np # type: ignore
from lib.misc import CropBox
from PIL import Image
import manga_ocr # type: ignore
from lib.types.preset import Preset

class Bubble:
	def __init__(self, img:cv2.typing.MatLike, box:CropBox, mocr:manga_ocr.MangaOcr, preset:Preset) -> None:
		self.img:cv2.typing.MatLike = box.crop(img)
		self.box:CropBox = box
		self.preset:Preset = preset
		self.mocr:manga_ocr.MangaOcr = mocr
		self.area:int = box.w*box.h
		# self.mask
		# self.mask:cv2.typing.MatLike = self.createMask(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) if len(self.img.shape) > 2 else self.img)
		self.hasContent:bool = self.checkForContent()
		# self.keyPoints:list[cv2.typing.MatLike] = []
		# self.text:str = ""
		# self.translation:str|None = None
		
	# def translate(self) -> str|None:
	# 	if self.hasContent:
	# 		Image.fromarray(self.img).save(f"./out/p_{self.area}.jpg", optimize=True) # type: ignore
	# 		self.text:str = self.mocr(Image.fromarray(self.img)) # type: ignore
	# 		lt = deep_translator.MyMemoryTranslator(source="ja-JP", target="en-GB")
	# 		self.translation = lt.translate(self.text) # type: ignore
	# 		print(self.text)
		
	def createMask(self, img:cv2.typing.MatLike) -> cv2.typing.MatLike:
		threshed:cv2.typing.MatLike = cv2.threshold(cv2.medianBlur(img, self.preset.maskBlur), self.preset.maskThresh, 255, cv2.THRESH_BINARY)[1] # // reduced threshold less black
		morphed:cv2.typing.MatLike = cv2.morphologyEx(threshed, cv2.MORPH_RECT, cv2.getStructuringElement(cv2.MORPH_RECT,self.preset.maskMorph)) # // (9,9) < -> less thick > -> thicker
		cMorphed = cv2.cvtColor(morphed, cv2.COLOR_GRAY2RGB)
		# contours:list[cv2.typing.MatLike] = cv2.findContours(morphed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0] # type: ignore
		# cv2.drawContours(cMorphed,contours,-1,(0,255,255),2)
		# agast = cv2.AgastFeatureDetector()
		# agast.create(100,False,cv2.AGAST_FEATURE_DETECTOR_OAST_9_16)
		# points = agast.detect(morphed)
		# cMorphed = cv2.drawKeypoints(cMorphed,points)
		Image.fromarray(cMorphed).save(f"./out/b/m/{self.area}.jpg")
		# filteredContours:list[cv2.typing.MatLike] = [contour for contour in contours if cv2.contourArea(contour) > self.preset.maskContourFilterMaxArea*self.area and cv2.mean(morphed, mask=cv2.bitwise_not(morphed))[0] == 0]
		# mask:cv2.typing.MatLike = np.zeros_like(morphed)
		# cv2.drawContours(mask, filteredContours, -1, 1, thickness=cv2.FILLED) # type: ignore
		morphed = cv2.bitwise_not(morphed)
		return morphed
		# return cv2.bitwise_and(morphed, morphed, mask=mask)
		
	def checkForContent(self) -> bool:
		# Image.fromarray(self.mask)
		return True
		blurred:cv2.typing.MatLike = cv2.GaussianBlur(self.img, (3,3), 1) # type: ignore // sigma 0, 1 or 2
		keypoints = cv2.AgastFeatureDetector_create( # type: ignore
			threshold=120,
			nonmaxSuppression=False,
			type=cv2.AGAST_FEATURE_DETECTOR_OAST_9_16
		).detect(blurred)
		self.keyPoints = keypoints
		blurred = cv2.drawKeypoints(blurred,keypoints, None, color=(0,255,0)) # type: ignore
		Image.fromarray(blurred).save(f"./out/p_{self.area}.jpg") # type: ignore
		# return self.area/len(keypoints) < 150 if len(keypoints) > 0 else False
		return self.area/len(keypoints) < 100 if len(keypoints) > 0 else False # type: ignore