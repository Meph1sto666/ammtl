from lib.types.bubble import Bubble
from lib.types.page import Page
import tkinter as tk
from PIL import ImageTk, Image

class PagePreview:
	def __init__(self, root:tk.Tk) -> None:
		self.rectangles:list[list[int|tuple[int,int,int,int]|bool]] = []

		image:Image = Image.open("tests/4.jpg") # type: ignore
		self.image_tk = ImageTk.PhotoImage(image) # type: ignore
		self.canvas = tk.Canvas(root, width=self.image_tk.width(), height=self.image_tk.height(), bg="black")
		self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		
		self.canvas.configure(scrollregion=self.canvas.bbox("all"))
		self.canvas.create_image(0, 0, anchor='nw', image=self.image_tk) # type: ignore
		self.canvas.bind('<Motion>', self.update) # type:ignore
		self.canvas.bind("<Button-1>", self.handleLeft) # type:ignore
		self.canvas.bind_all("<Control-z>" ,self.delLast) # type:ignore

	def update(self, event:tk.Event) -> None:
		if len(self.rectangles) < 1: return
		if self.rectangles[-1][1]: return
		x1:int = int(self.rectangles[-1][2][0]) if self.rectangles[-1][2][0] < event.x else event.x
		y1:int = int(self.rectangles[-1][2][1]) if self.rectangles[-1][2][1] < event.y else event.y
		x2:int = int(self.rectangles[-1][2][0]) if self.rectangles[-1][2][0] > event.x else event.x
		y2:int = int(self.rectangles[-1][2][1]) if self.rectangles[-1][2][1] > event.y else event.y
		self.canvas.coords(self.rectangles[-1][0], x1, y1, x2, y2)

	def handleLeft(self, event:tk.Event) -> None:
		if not self.rectangles[-1][1] if len(self.rectangles) > 0 else False:
			self.rectangles[-1][1] = True
			return
		rec:int = self.canvas.create_rectangle(event.x, event.y, event.x+50, event.y+50, outline="red")
		self.rectangles.append([rec, False, (event.x, event.y, event.x, event.y)])

	def delLast(self, event) -> None:
		if len(self.rectangles) < 1: return
		self.canvas.delete(self.rectangles[-1][0])
		self.rectangles.pop()


root:tk.Tk = tk.Tk()
exit = tk.Button(root, text="Exit", command=root.destroy)
exit.pack()
pp = PagePreview(root)
root.mainloop()