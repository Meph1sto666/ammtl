import os
import pstats
import cv2
from PIL import Image # type: ignore
from lib.types.page import Page
from lib.types.preset import Preset
# import pytesseract
# tsr = pytesseract.pytesseract.tesseract_cmd = "./dep/Tesseract-OCR/tesseract.exe"
# import manga_ocr
# mocr = manga_ocr.MangaOcr()
mocr = None

import cProfile
profiler = cProfile.Profile()
profiler.enable()
for f in os.listdir("./tests/")[0:]:
	p = Page(cv2.imread(f"./tests/{f}"), mocr, Preset()) # type: ignore
	# diff = p.tTracker.diff()
	# dStr = f"{colorama.Back.RESET+f.ljust(10)}{colorama.Back.RESET}: {(colorama.Fore.RESET+' / ').join([timeToColorPrec(diff[d])+str(d)+' ['+str(round(diff[d],2)).split('.')[0].rjust(4)+'.'+str(round(diff[d],2)).split('.')[1].ljust(2)+']' for d in diff])}"
	# print(dStr)

	# Image.fromarray(p.out).save(f"./out/{f}") # type: ignore
	# Image.fromarray(p.mask).save(f"./out/mask_{f}") # type: ignore
profiler.disable()
profiler.dump_stats("./out/stats.dmp")

with open("./out/stats.txt", "w") as f:
    ps = pstats.Stats("./out/stats.dmp", stream=f)
    # ps.sort_stats('time')
    ps.sort_stats('cumulative')
    # ps.sort_stats('ncalls')
    ps.print_stats()