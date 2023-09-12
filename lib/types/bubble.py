import cv2
import cv2.typing
import deep_translator
from lib.misc import CropBox
from PIL import Image
import manga_ocr # type: ignore

class Bubble:
	def __init__(self, img:cv2.typing.MatLike, box:CropBox, mocr:manga_ocr.MangaOcr) -> None:
		self.box:CropBox = box
		self.img:cv2.typing.MatLike = box.crop(img)
		self.area:int = box.w*box.h
		self.keyPoints:list[cv2.typing.MatLike] = []
		self.mocr:manga_ocr.MangaOcr = mocr
		self.hasContent:bool = self.checkForContent()
		self.text:str = ""
		
		self.translation:str|None = None
		
	def translate(self) -> str|None:
		if self.hasContent:
			Image.fromarray(self.img).save(f"./out/p_{self.area}.jpg", optimize=True) # type: ignore
			self.text:str = self.mocr(Image.fromarray(self.img)) # type: ignore
			lt = deep_translator.MyMemoryTranslator(source="ja-JP", target="en-GB")
			self.translation = lt.translate(self.text) # type: ignore
			print(self.text)
		
	def checkForContent(self) -> bool:
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