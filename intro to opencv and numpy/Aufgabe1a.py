import numpy as np
import cv2

img = cv2.imread("PATH to KITTI46_13.png", 1)
b = img.copy()
b[:, :, 1] = 0
b[:, :, 2] = 0

g = img.copy()
g[:, :, 0] = 0
g[:, :, 2] = 0

r = img.copy()
r[:, :, 0] = 0
r[:, :, 1] = 0

cv2.imshow('BLUE', b)

cv2.imshow('GREEN', g)

cv2.imshow('RED', r)

cv2.waitKey(0)
