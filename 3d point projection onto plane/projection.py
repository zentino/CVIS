# Aufgabe1: Punktwolken-Projektion
import numpy as np
import cv2

fx = 460
fy = 460
cx = 320
cy = 240

X1 = np.array([10., 10., 100.])
X2 = np.array([33., 22., 111.])
X3 = np.array([100., 100., 1000.])
X4 = np.array([20, -100, 100])

# Verschiebungsvektor
t = np.zeros((3, 1))
# Rotationsmatrix
R = np.eye(3)
Rt = np.hstack((R, t))

# Kalibrierungsmatrix
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
P = K.dot(Rt)

x1 = P.dot(np.hstack((X1, 1)))
x2 = P.dot(np.hstack((X2, 1)))
x3 = P.dot(np.hstack((X3, 1)))
x4 = P.dot(np.hstack((X4, 1)))

# Durch den letzten Wert teilen ([:2) -> entfernt den letzten Wert aus der liste)
x1 = x1[:2] / x1[2]
x2 = x2[:2] / x2[2]
x3 = x3[:2] / x3[2]
x4 = x4[:2] / x4[2]

print(x1)
print(x2)
print(x3)
print(x4)

points3D = np.array([X1, X2 ,X3, X4])
points2D, jacobian = cv2.projectPoints(points3D, R, t, K, np.array([], dtype=float))

print(points2D)
#?
print(jacobian)
