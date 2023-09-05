import cv2
from PIL import Image
import os

for f in os.listdir("./tests/"):
    img = cv2.imread(f"./tests/{f}")
    keypoints = cv2.AgastFeatureDetector_create(
                threshold=100,
                nonmaxSuppression=False,
                type=cv2.AGAST_FEATURE_DETECTOR_OAST_9_16
            ).detect(img)

    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0)) # type: ignore
    Image.fromarray(img_with_keypoints).save(f"./out/{f}") # type: ignore