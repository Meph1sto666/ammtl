import cv2.typing
import cv2
import numpy as np
from PIL import Image

img = cv2.imread("./tests/4.jpg")
area:int = img.shape[0]*img.shape[1]
gray:cv2.typing.MatLike = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
maskThresh:int = 0
maskBlur:int = 3
maskMorph: tuple[int, int] = (7,7)
maskContourFilterMaxArea:float = .0058
def createMask() -> cv2.typing.MatLike:
	blurred:cv2.typing.MatLike = cv2.medianBlur(gray, maskBlur)
	threshed:cv2.typing.MatLike = cv2.threshold(blurred, maskThresh, 255, cv2.THRESH_BINARY)[1] # // reduced threshold less black
	morphed:cv2.typing.MatLike = cv2.morphologyEx(threshed, cv2.MORPH_RECT, cv2.getStructuringElement(cv2.MORPH_RECT,maskMorph)) # // (9,9) < -> less thick > -> thicker
	contours:list[cv2.typing.MatLike] = cv2.findContours(morphed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0] # type: ignore
	meanS:float = cv2.mean(morphed, mask=cv2.bitwise_not(morphed))[0]
	filteredContours:list[cv2.typing.MatLike] = [contour for contour in contours if cv2.contourArea(contour) > maskContourFilterMaxArea*area and meanS == 0]
	mask:cv2.typing.MatLike = np.zeros_like(morphed)
	cv2.drawContours(mask, filteredContours, -1, 1, thickness=cv2.FILLED) # type: ignore
	morphed = cv2.bitwise_not(morphed) # type: ignore
	return cv2.bitwise_and(morphed, morphed, mask=mask)

for i in range(500, 600, 10):
	maskContourFilterMaxArea = i/100000
	for j in range(220, 241):
		maskThresh = j
		Image.fromarray(createMask()).save(f"./out/mask_test/4_i{i}j{j}.jpg")
