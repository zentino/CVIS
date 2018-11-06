import numpy as np
import cv2
import glob

# Pfade aller Bilder mit 640 x 360 Auflösung einlesen
image_paths = glob.glob("calib_images/*.jpg")
# Arrays für erkannte Ecken im Bild und 3D Punkte
imagePoints = []
objectPoints = []
images = []

objp = np.zeros((9*6, 3), dtype=np.float32)
# [:,:2] -> in den ersten zwei Spalten einfügen (Z Koordinate bleibt 0)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

for path in image_paths:
    # 1. Kalibrierungsbilder einlesen
    image = cv2.imread(path)
    img = image.copy()
    ret_val, corners = cv2.findChessboardCorners(image, (9, 6), None)
    # 2. Wurde Muster gefunden?
    if ret_val:
        images.append(img)
        # hinzufügen der Bildpunkte und korrespondierende 3D Punkte wenn Muster erkannt
        imagePoints.append(corners)
        objectPoints.append(objp)

        # 3. Die erkannten Ecken im Bild anzeigen
        image = cv2.drawChessboardCorners(image, (9, 6), corners, ret_val)
        cv2.imshow('Chessboard corners', image)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# 4. Kamera Kalibrierung
ret, cM, dC, rVecs, tVecs = cv2.calibrateCamera(objectPoints, imagePoints, (9, 6), None, None)
print(cM)
i = 0
"""5. Die 3D Punkte der Schachbrettecken mit Hilfe der bestimmten Kameraparameter 
zurück ins Bild projiziert und anzeigt"""
for image in images:
    points2D, jacobian = cv2.projectPoints(objp, rVecs[i], tVecs[i], cM, dC)
    img = cv2.drawChessboardCorners(image, (9, 6), points2D, True)
    cv2.imshow('Chessboard corners', img)
    cv2.waitKey(500)
    i = i + 1
