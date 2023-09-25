import cv2.typing
import cv2
import numpy as np
from PIL import Image

img = cv2.imread("./tests/4.jpg")
area:int = img.shape[0]*img.shape[1]
gray:cv2.typing.MatLike = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
maskThresh:int = 236
maskBlur:int = 7
maskMorph: tuple[int, int] = (9,9)
maskContourFilterMinArea:float = .0058
maskContourFilterMaxArea:float = .02

def createMask() -> cv2.typing.MatLike:
	mask = gray
	mask = cv2.bitwise_and(gray,gray,mask=cv2.bitwise_not(gray))
	mask:cv2.typing.MatLike = cv2.stackBlur(mask, (9,9))
	mask:cv2.typing.MatLike = cv2.medianBlur(mask, 19)
	mask = cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_DILATE,(3,3)))
	mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
	return cv2.bitwise_not(mask)

i = 0
# Image.fromarray(createMask()).save(f"./out/mask_test/4_i{i}.jpg")

masked = cv2.bitwise_and(gray,gray,mask=createMask())
cnts = cv2.findContours(masked,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]
masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2RGB)
masked = cv2.drawContours(masked, cnts, -1, (0,255,0), 2)
Image.fromarray(masked).save(f"./out/mask_test/4_i{i}.jpg")