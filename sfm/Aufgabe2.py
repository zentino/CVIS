import numpy as np
import cv2

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
end_header 
'''


def write_ply(fn, verts):
    verts = verts.reshape(-1, 3)
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f')


def verify_camerapose(P0,K, R1, R2, t, pts1, pts2):

    P1 = K.dot(np.hstack((R1, t)))
    P2 = K.dot(np.hstack((R1, -t)))
    P3 = K.dot(np.hstack((R2, t)))
    P4 = K.dot(np.hstack((R2, -t)))
    cMs = [P1, P2, P3, P4]
    list = []
    for p in cMs:
        X = cv2.triangulatePoints(P0,p,np.array(pts1,dtype=np.float),np.array(pts2,dtype=np.float))
        X /= X[3]
        list.append(check_3DPoints(X))

    max_value = max(list)
    max_index = list.index(max_value)
    return cMs[max_index]


def check_3DPoints(objectPoints):
    countPositiveDepth = 0
    for o in objectPoints.T:
        print(o)
        print(o[2])
        if o[2] >= 0:
            countPositiveDepth += 1

    return countPositiveDepth


def reconstruction():
    F = np.genfromtxt("FMatrix.txt",dtype=float,delimiter=',')
    pts1 = np.loadtxt("pts1.txt",dtype=int, delimiter=',')
    pts2 = np.loadtxt("pts2.txt",dtype=int, delimiter=',')

    fx = fy = 721.5
    cx = 690.5
    cy = 172.8
    K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
    Rt = np.hstack((np.eye(3), np.zeros((3, 1))))
    P0 = K.dot(Rt)
    E = K.T * np.mat(F) * K
    R1, R2, t = cv2.decomposeEssentialMat(E)
    P1 = verify_camerapose(P0, K, R1, R2, t, pts1.reshape(-1, 1, 2), pts2.reshape(-1, 1, 2))
    pointcloud = cv2.triangulatePoints(P0, P1, np.array(pts1.reshape(-1, 1, 2),dtype=np.float), np.array(pts2.reshape(-1, 1, 2),dtype=np.float))
    write_ply("KITTI11" + 'punktwolke.ply', pointcloud)


reconstruction()
