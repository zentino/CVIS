import numpy as np
import cv2

img = cv2.imread("PATH to KITTI46_13.png", 1)

# b, g und r sind Graustufenbilder
b, g, r = cv2.split(img)
zeros = b * 0
# Um ein blaues Bild anzuzeigen, müssen wir die Farbkanäle 1 und 2 (Grün und Rot) auf 0 setzen
b = cv2.merge((b, zeros, zeros))
g = cv2.merge((zeros, g, zeros))
r = cv2.merge((zeros, zeros, r))
# cv2.imwrite("Blau.png", b)
# cv2.imwrite("Rot.png", b)
# cv2.imwrite("Gruen.png", b)
cv2.imshow("b", b)
cv2.imshow("g", g)
cv2.imshow("r", r)

cv2.waitKey(0)
