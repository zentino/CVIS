import numpy as np
import cv2
import glob

image_paths = glob.glob("calib_images/*.jpg")
imagePoints = []
objectPoints = []
images = []

objp = np.zeros((9 * 6, 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

for path in image_paths:
    image = cv2.imread(path)
    img = image.copy()
    ret_val, corners = cv2.findChessboardCorners(image, (9, 6), None)
    if ret_val:
        images.append(img)
        imagePoints.append(corners)
        objectPoints.append(objp)

ret, cM, dC, rVecs, tVecs = cv2.calibrateCamera(objectPoints, imagePoints, (9, 6), None, None)
print("Camera matrix")
print(cM)

def show_img(points_3d, cameraMatrix, distCoeffs, rvecs, tvecs, imagesList):
    fx = float(input("Bitte geben Sie einen Wert für die Variable fx > "))
    fy = float(input("Bitte geben Sie einen Wert für die Variable fy > "))
    cx = float(input("Bitte geben Sie einen Wert für die Variable cx > "))
    cy = float(input("Bitte geben Sie einen Wert für die Variable cy > "))
    cameraMatrix[0][0] = fx
    cameraMatrix[0][2] = cx
    cameraMatrix[1][1] = fy
    cameraMatrix[1][2] = cy

    i = 0
    for im in imagesList:
        points_2d, jacobian = cv2.projectPoints(points_3d, rvecs[i], tvecs[i], cameraMatrix, None)
        cv2.imshow('Chessboard corners', cv2.drawChessboardCorners(im.copy(), (9,6), points_2d, True))
        cv2.waitKey(500)
        i = i + 1

    cv2.destroyAllWindows()
    repeat = input("Wiederholen? y/n")
    if repeat == "y":
        show_img(points_3d, cameraMatrix, distCoeffs, rvecs, tvecs, images)
    else:
        print("Bye")


show_img(objp, cM, dC, rVecs, tVecs, images)
