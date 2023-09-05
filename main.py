import os
import cv2
from PIL import Image
from lib.types.page import Page
# import pytesseract
# tsr = pytesseract.pytesseract.tesseract_cmd = "./dep/Tesseract-OCR/tesseract.exe"
import manga_ocr
# mocr = manga_ocr.MangaOcr()
mocr = None

for f in os.listdir("./tests/"):
    p = Page(cv2.imread(f"./tests/{f}"), mocr)
    Image.fromarray(p.img).save(f"./out/{f}") # type: ignore