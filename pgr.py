import json
from lib.types.bubble import Bubble # type: ignore
from lib.types.page import Page # type: ignore
import tkinter as tk
from PIL import ImageTk, Image

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
		

root:tk.Tk = tk.Tk()
pp = PagePreview(root, "./tests/007.jpg")
root.mainloop()
