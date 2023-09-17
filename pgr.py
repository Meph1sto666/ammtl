import json
import typing
import numpy as np
import cv2
from lib.types.bubble import Bubble
from lib.types.page import Page # type: ignore
from lib.types.preset import Preset
import tkinter as tk
from PIL import ImageTk, Image
import os
import difflib
from datetime import datetime as dt
import cv2.typing
import random

class PagePreview:
	def __init__(self, root:tk.Tk, page:str) -> None:
		self.page:str = page
		self.rectangles:list[list[int|tuple[int,int,int,int]|bool]] = []
		self.savedCoords:list[tuple[int, ...]] = []
		self.root = root
		image:Image = Image.open(page) # type: ignore
		self.image_tk = ImageTk.PhotoImage(image) # type: ignore
		self.canvas = tk.Canvas(root, width=self.image_tk.width(), height=self.image_tk.height(), bg="black")
		self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
		self.canvas.configure(scrollregion=self.canvas.bbox("all"))

		scrollbar = tk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.canvas.yview) # type: ignore
		self.canvas.configure(yscrollcommand=scrollbar.set)

		# ===bindings===
		self.canvas.create_image(0, 0, anchor='nw', image=self.image_tk) # type: ignore
		self.canvas.bind('<Motion>', self.update) # type: ignore
		self.canvas.bind("<Button-1>", self.handleLeft) # type: ignore
		self.canvas.bind_all("<Control-z>", self.delLast) # type: ignore
		self.canvas.bind_all("<Control-s>", self.saveRecs) # type: ignore
		self.canvas.bind("<MouseWheel>", self.scroll_canvas) # type: ignore

	def scroll_canvas(self, event:tk.Event): # type: ignore
		if (int(event.state)-8) == 1:
			self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")  # Horizontal scroll
		else:
			self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")  # Vertical scroll

	def update(self, event:tk.Event) -> None: # type: ignore
		if len(self.rectangles) < 1: return
		if self.rectangles[-1][1]: return
		ex = event.widget.canvasx(event.x) # type: ignore
		ey = event.widget.canvasy(event.y) # type: ignore
		x1:int = (int(self.rectangles[-1][2][0]) if self.rectangles[-1][2][0] < ex else ex) # type: ignore
		y1:int = (int(self.rectangles[-1][2][1]) if self.rectangles[-1][2][1] < ey else ey) # type: ignore
		x2:int = (int(self.rectangles[-1][2][0]) if self.rectangles[-1][2][0] > ex else ex) # type: ignore
		y2:int = (int(self.rectangles[-1][2][1]) if self.rectangles[-1][2][1] > ey else ey) # type: ignore
		self.canvas.coords(self.rectangles[-1][0], x1, y1, x2, y2) # type: ignore

	def handleLeft(self, event:tk.Event) -> None: # type: ignore
		if not self.rectangles[-1][1] if len(self.rectangles) > 0 else False:
			self.rectangles[-1][1] = True
			return
		ex = event.widget.canvasx(event.x) # type: ignore
		ey = event.widget.canvasy(event.y) # type: ignore
		rec:int = self.canvas.create_rectangle(ex, ey, ex, ey, outline="red") # type: ignore
		self.rectangles.append([rec, False, (ex, ey, ex, ey)])

	def delLast(self, event) -> None: # type: ignore
		if len(self.rectangles) < 1: return
		self.canvas.delete(self.rectangles[-1][0]) # type: ignore
		self.rectangles.pop()

	def saveRecs(self, event:tk.Event) -> None: # type: ignore
		self.savedCoords = [tuple(self.canvas.coords(r[0])) for r in self.rectangles] # type: ignore
		json.dump(self.savedCoords, open(f"./presets/traindata/{self.page.split('/')[-1].split('.')[0]}.json", "w", encoding="utf-8"))
		

# root:tk.Tk = tk.Tk()
# pp = PagePreview(root, "./tests/4.jpg")
# root.mainloop()

imgFile:str = "./tests/4.jpg"

def testPresets():
	preset:Preset = Preset()
	p:Page = Page(cv2.imread(imgFile), None, preset)
	cmbs:list[tuple[Preset, list[Bubble]]] = []
	c:int = 0
	last:dt = dt.now()
	start:dt = dt.now()
	for maskThresh in range(230, 240, 1):
		preset.maskThresh = maskThresh
		for conjectionBlur in range(3,9,2):
			preset.conjectionBlur = conjectionBlur
			for conjectionThresh in range(200, 210, 1):
				preset.conjectionThresh = conjectionThresh
				for conjectionClusterEps in range(300, 400, 10):
					preset.conjectionClusterEps = conjectionClusterEps/10000
					p.update()
					coords = []
					for b in p.bubbles:
						box = b.box.toTuple()
						coords.append((box[0], box[1], box[0]+box[2], box[1]+box[3]))
					cmbs.append((preset.__dict__, coords))
					c+=1
					if c%50==0:
						print(c, (dt.now()-last).total_seconds(), end=" / ")
						last = dt.now()
						json.dump(cmbs, open(f"./presets/traindata/found/{imgFile.split('/')[-1].split('.')[0]}_{c}.json","w"))
						cmbs = []
	if len(cmbs) > 0:
		json.dump(cmbs, open(f"./presets/traindata/found/{imgFile.split('/')[-1].split('.')[0]}_{c}.json","w"))
	print(f"TOTAL: {(dt.now()-start).total_seconds()}")
# testPresets()

sample:list[list[int]] = json.load(open("./presets/traindata/4.json"))
foundFolder:str = "./presets/traindata/found/"
diffFolder:str = "./presets/traindata/diff/"
groupedFolder:str = "./presets/traindata/grouped/"

sortedSample:list[list[int]] = sorted(sample, key=lambda x: (x[0], x[1], x[2], x[3]))
diffed:list[tuple[float, dict[str, int|float|str], list[tuple[int,int,int,int]]]] = []
for file in list(filter(lambda x: x.endswith(".json"), os.listdir(foundFolder))):
	found:list[tuple[dict[str, int|float|str],list[tuple[int,int,int,int]]]] = json.load(open(f"{foundFolder}{file}"))
	for f in range(len(found)):
		sortedFound:list[tuple[int, int, int, int]] = sorted(found[f][1], key=lambda x: (x[0], x[1], x[2], x[3]))
		seq = difflib.SequenceMatcher(None, str(sample), str(sortedFound))
		diffed.append((seq.ratio(), *found[f]))
diffed = sorted(diffed, key=lambda x: x[0], reverse=True)
json.dump(diffed, open(f"./presets/traindata/diffed/{imgFile.split('/')[-1].split('.')[0]}.json", "w"))

img: cv2.typing.MatLike = cv2.imread(imgFile)
for s in sample:
	cv2.rectangle(img, pt1=(int(s[0]),int(s[1])), pt2=(int(s[2]), int(s[3])), color=(0,255,255), thickness=2)
for d in diffed[0:1]:#[int(len(diffed)/2):int(len(diffed)/2)+1]:
	# print(d[2])
	color = (random.randint(1,256),random.randint(1,256),random.randint(1,256))
	for c in d[2]:
		cv2.rectangle(img, (c[0],c[1]), (c[2],c[3]), color, 2)
Image.fromarray(img).show() # type: ignore