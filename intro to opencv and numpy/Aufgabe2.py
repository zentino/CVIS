import numpy as np
import cv2

img = cv2.imread("PATH to KITTI46_13.png", 1)

region = img[170:270, 790:1070]
#(ValueError: could not broadcast input array from shape (100,280,3) into shape (100,210,3))
img[170:270, 490: 770] = region
# cv2.imwrite(...)
cv2.imshow("Region", img)
cv2.waitKey(0)