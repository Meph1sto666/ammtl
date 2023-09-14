import colorama
import cv2
import cv2.typing
from datetime import datetime as dt, timedelta
colorama.init(True)

class Delta:
	def __init__(self, description:str|None=None, time:dt|None=None, group:str|None=None) -> None:
		self.TIME:dt = time if time != None else dt.now()
		self.description:str|None = description
		self.group = group

	def __sub__(self, other) -> timedelta: # type: ignore
		if isinstance(other, Delta):
			return (self.TIME-other.TIME)
		else: raise TypeError("Unsupported operand type: {}".format(type(other))) # type: ignore

class TimeTracker:
	def __init__(self, startTime:dt) -> None:
		self.START_TIME:dt = startTime
		self.deltas:list[Delta] = []

	def add(self, d:Delta) -> None:
		self.deltas.append(d)

	def diff(self, ms:bool=True) -> dict[str|None, float]:#list[tuple[str|None, float]]:
		if len(self.deltas) < 1: return {}
		grouped:dict[str|None, list[Delta]] = {}
		for d in self.deltas:
			if grouped.get(d.group):
				grouped[d.group].append(d)
			else:
				grouped[d.group] = [d]
		diffed:dict[str|None, float] = {}
		for g in grouped:
			total:float = .0
			for d in range(len(grouped[g])):
				if d == 0:
					i = self.deltas.index(grouped[g][0])
					t = ((self.deltas[i].TIME-self.START_TIME) if i==0 else (self.deltas[i]-self.deltas[i-1]))
				else:
					t = (grouped[g][d]-grouped[g][d-1])
				diffed[grouped[g][d].description] = t.total_seconds()*(1000 if ms else 1)
				total+=t.total_seconds()*(1000 if ms else 1)
			if g!=None:
				diffed[str(grouped[g][0].group)+"_TOTAL"] = total
		# diffed = dict([(self.deltas[d].description, (self.deltas[d].TIME - self.START_TIME if d == 0 else self.deltas[d].TIME - self.deltas[d-1].TIME).total_seconds()*1000) for d in range(len(self.deltas))])
		diffed["TOTAL"] = (self.deltas[-1].TIME - self.START_TIME).total_seconds()*(1000 if ms else 1)
		return diffed

class CropBox:
	def __init__(self, x:int, y:int, w:int, h:int, tolerance:int) -> None:
		self.x:int = x
		self.y:int = y
		self.w:int = w
		self.h:int = h
		self.tolerance:int = tolerance

	def crop(self, img:cv2.typing.MatLike, tolerance:int|None=None) -> cv2.typing.MatLike:
		tolerance = self.tolerance if tolerance==None else tolerance
		return img[max(0,self.y-tolerance):min(img.shape[0]-1,self.y+self.h+tolerance), max(0,self.x-tolerance):min(img.shape[1]-1,self.x+self.w+tolerance)] # type: ignore

	def toTuple(self, tolerance:int=0) -> tuple[int, int, int, int]:
		return (self.x-tolerance, self.y-tolerance, self.w+tolerance*2, self.h+tolerance*2)

def drawBoundingBox(image:cv2.typing.MatLike,x:int,y:int,w:int,h:int,text:str|None=None, inset:bool=False, color:tuple[int,int,int]=(0,255,0)) -> cv2.typing.MatLike:
	"""Draws a bounding box in an image

	Args:
		image (cv2.typing.MatLike): Image to draw on
		x (int): x pos
		y (int): y pos
		w (int): Rectangle width
		h (int): Rectangle height
		text (str | None, optional): Text to write to the rect. Defaults to None.
		inset (bool, optional): If true the text will be inside the rectangle else outside. Defaults to False.

	Returns:
		cv2.typing.MatLike: Image with drawn boxes
	"""
	# color:tuple[int,...] = (random.randint(1,256),random.randint(1,256),random.randint(1,256))
	cv2.rectangle(image, (x, y), (x + w, y + h), color, 2) # type: ignore
	if text != None:
		cv2.putText(image, text, (x if not inset else x+5, (y - 10) if not inset else (y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA) # type: ignore
	return image

def timeToColorPrec(t:float, ms:bool=True) -> str:
	"""Converts the time to a colorama string for easier speed recognition on console output (more precise; 0-500ms)

	Args:
		t (float): Taken time

	Returns:
		str: Corresponding color string
	"""
	c:str = colorama.Fore.RESET
	if   t < .050*(1000 if ms else 1): c = colorama.Fore.WHITE
	elif t < .100*(1000 if ms else 1): c = colorama.Fore.LIGHTMAGENTA_EX
	elif t < .150*(1000 if ms else 1): c = colorama.Fore.MAGENTA
	elif t < .200*(1000 if ms else 1): c = colorama.Fore.LIGHTCYAN_EX
	elif t < .250*(1000 if ms else 1): c = colorama.Fore.CYAN
	elif t < .300*(1000 if ms else 1): c = colorama.Fore.LIGHTGREEN_EX
	elif t < .350*(1000 if ms else 1): c = colorama.Fore.GREEN
	elif t < .400*(1000 if ms else 1): c = colorama.Fore.LIGHTYELLOW_EX
	elif t < .450*(1000 if ms else 1): c = colorama.Fore.YELLOW
	elif t < .500*(1000 if ms else 1): c = colorama.Fore.LIGHTRED_EX
	else: c = colorama.Fore.RED
	return c