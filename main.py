import os
import cv2
from PIL import Image
from lib.types.page import Page
# import pytesseract
# tsr = pytesseract.pytesseract.tesseract_cmd = "./dep/Tesseract-OCR/tesseract.exe"
# import manga_ocr
# mocr = manga_ocr.MangaOcr()
mocr = None

for f in os.listdir("./tests/"):
    print(f, end=" ")
    p = Page(cv2.imread(f"./tests/{f}"), mocr) # type: ignore
    print(f"t={p.tDelta}ms")
    Image.fromarray(p.img).save(f"./out/{f}") # type: ignore
    Image.fromarray(p.mask).save(f"./out/mask_{f}") # type: ignore