import cv2

class CropBox:
	def __init__(self, x:int, y:int, w:int, h:int, tolerance:int) -> None:
		self.x:int = x
		self.y:int = y
		self.w:int = w
		self.h:int = h
		self.tolerance:int = tolerance

	def crop(self, img:cv2.Mat, tolerance:int|None=None) -> cv2.Mat:
		tolerance = self.tolerance if tolerance==None else tolerance
		return img[max(0,self.y-tolerance):min(img.shape[0]-1,self.y+self.h+tolerance), max(0,self.x-tolerance):min(img.shape[1]-1,self.x+self.w+tolerance)]

	def toTuple(self, tolerance:int=0) -> tuple[int, int, int, int]:
		return (self.x-tolerance, self.y-tolerance, self.w+tolerance*2, self.h+tolerance*2)

def drawBoundingBox(image:cv2.Mat,x:int,y:int,w:int,h:int,text:str|None=None, inset:bool=False) -> cv2.Mat:
	"""Draws a bounding box in an image

	Args:
		image (cv2.Mat): Image to draw on
		x (int): x pos
		y (int): y pos
		w (int): Rectangle width
		h (int): Rectangle height
		text (str | None, optional): Text to write to the rect. Defaults to None.
		inset (bool, optional): If true the text will be inside the rectangle else outside. Defaults to False.

	Returns:
		cv2.Mat: Image with drawn boxes
	"""
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) # type: ignore
	if text != None:
		cv2.putText(image, text, (x if not inset else x+5, (y - 10) if not inset else (y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA) # type: ignore
	return image