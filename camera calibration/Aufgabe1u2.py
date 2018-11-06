import numpy as np
import cv2
import glob

image_pathes = glob.glob("calib_images/*.jpg")
# Arrays für erkannte Ecken im Bild und 3D Punkte
imagePoints = []
objectPoints = []

objp = np.zeros((9 * 6, 3), np.float32)
# [:,:2] -> in den ersten zwei Spalten einfügen (Z Koordinate bleibt 0)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

for path in image_pathes:
    # 1. Kalibrierungsbilder einlesen
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret_val, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    # 2. Wurde Muster gefunden?
    if ret_val:
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

"""5. Die 3D Punkte der Schachbrettecken mit Hilfe der bestimmten Kameraparameter 
zurück ins Bild projiziert und anzeigt"""
def project_points(points_3d, cameraMatrix, distCoeffs, rvecs, tvecs, imageSize):
    i = 0
    for path in image_pathes:
        im = cv2.imread(path)
        points_2d, jacobian = cv2.projectPoints(points_3d, rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
        im = cv2.drawChessboardCorners(im, imageSize, points_2d, True)
        cv2.imshow('Chessboard corners', im)
        cv2.waitKey(500)
        i = i + 1
    cv2.destroyAllWindows()


project_points(objp, cM, dC, rVecs, tVecs,(9,6))


# -------------------------------------------------------
# "Aufgabe 2: Modifizierte 3D Punktprojektion"
print()
cameraMatrix = cM
fx = float(input("Bitte geben Sie einen Wert für die Variable fx > "))
fy = float(input("Bitte geben Sie einen Wert für die Variable fy > "))
cx = float(input("Bitte geben Sie einen Wert für die Variable cx > "))
cy = float(input("Bitte geben Sie einen Wert für die Variable cy > "))
cameraMatrix[0][0] = fx
cameraMatrix[0][2] = cx
cameraMatrix[1][1] = fy
cameraMatrix[1][2] = cy

project_points(objp, cameraMatrix, dC, rVecs, tVecs, (9,6))

repeat = input("Wiederholen? y/n")
if repeat == "y":
    project_points(objp, cameraMatrix, dC, rVecs, tVecs, (9,6))
else:
    print("Bye")
