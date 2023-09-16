import pickle
import typing

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
		self.conjectionClusterEps:float = 0.033 # uses img width // try
		self.conjectionClusterSamples:int = 2 # 1 or more // try a bit
		self.conjectionClusterAlgorithm:typing.Literal['auto', 'ball_tree', 'kd_tree', 'brute'] = "auto"
		self.conjectionClusterMinArea:float = 0.0003895
		self.conjectionClusterMaxArea:float = 0.0389553
		self.conjectionBubbleTolerance:int = 20 # only for the bubble borders

	def save(self, path:str) -> None:
		pickle.dump(self,open(path,"wb"))