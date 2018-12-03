# coding=utf-8
import numpy as np
import cv2
import glob

image_pathes = glob.glob('images/*.png')
fx = fy = 721.5
cx = 690.5
cy = 172.8
baseline = 0.54

ply_headerC = '''ply 
format ascii 1.0 
element vertex %(vert_num)d 
property float x 
property float y 
property float z 
property uchar red 
property uchar green 
property uchar blue 
end_header 
'''


def write_plyC(filename, verts):
    verts = verts.reshape(-1, 6)
    with open(filename, 'w') as f:
        f.write(ply_headerC % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')


def get_depth_img(f, tx, d):
    depth = np.zeros(d.shape)
    rows = d.shape[0]
    cols = d.shape[1]
    for x in range(0, rows):
        for y in range(0, cols):
            if d[x, y] == 0:
                depth[x, y] = 0
            else:
                depth[x, y] = (f * tx) / d[x, y]
    return depth


def get_pointcloud(img1, depth):
    rows = img1.shape[0]
    cols = img1.shape[1]
    vector = []
    for x in range(0, rows):
        for y in range(0, cols):
            if depth[x, y] != 0:
                Z = depth[x, y]
                u = x - cx
                v = y - cy
                X = (u * Z) / fx
                Y = (v * Z) / fx
                vector.append([X, Y, Z, img1[x, y][2], img1[x, y][1], img1[x, y][0]])
    return np.array(vector)


def get_transformed_pointcloud(t, pointcloud):
    for point in pointcloud:
        point[0] += t[0]
        point[1] += t[1]
        point[2] += t[2]
    return pointcloud


def project_pointcloud(pointcloud, img, fx, fy, cx ,cy):
    imgToReturn = np.zeros(img.shape, dtype=np.uint8)
    for point in pointcloud:
        X = point[0]
        Y = point[1]
        Z = point[2]
        i = np.floor((X*fx / Z) + cx)
        j = np.floor((Y*fy / Z) + cy)
        if i < imgToReturn.shape[0] and j < imgToReturn.shape[1]:
            imgToReturn[i, j] = np.array([point[5], point[4], point[3]])
    return imgToReturn


def dense_stereo_matching(img_pathes, fx, fy, cx, cy, baseline):
    blocksize = 12  # Gibt Fenstergröße an (1-20)
    min_disp = 1  # Gibt minimale Disparität an (0-10)
    y = 5
    num_disp = 16 * y  # Gibt maximale Disparität an (16 * y)
    img1 = cv2.imread(img_pathes[0], 1)
    img2 = cv2.imread(img_pathes[1], 1)

    stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=blocksize,
                                   speckleWindowSize=100, speckleRange=1)
    disparity = stereo.compute(img1, img2).astype(np.float32) / 16.0
    depth = get_depth_img(fx, baseline, disparity)

    t1 = np.array([0.3,0.3,0.3])
    t2 = np.array([0.6,0.6,0.6])
    t3 = np.array([0.9,0.9,0.9])
    t4 = np.array([1.2,1.2,1.2])
    pointcloud0 = get_pointcloud(img1,depth)
    pointcloud1 = get_transformed_pointcloud(t1, get_pointcloud(img1, depth))
    pointcloud2 = get_transformed_pointcloud(t2, get_pointcloud(img1, depth))
    pointcloud3 = get_transformed_pointcloud(t3, get_pointcloud(img1, depth))
    pointcloud4 = get_transformed_pointcloud(t4, get_pointcloud(img1, depth))

    write_plyC("pointclouds/pointscloud.ply", pointcloud0)
    write_plyC("pointclouds/pointscloud_transformed1.ply", pointcloud1)
    write_plyC("pointclouds/pointscloud_transformed2.ply", pointcloud2)
    write_plyC("pointclouds/pointscloud_transformed3.ply", pointcloud3)
    write_plyC("pointclouds/pointscloud_transformed4.ply", pointcloud4)
    img = project_pointcloud(pointcloud0, img1, fx, fy, cx, cy)
    cv2.imwrite("reprojection/img_pointcloud0.png", img)
    img = project_pointcloud(pointcloud1,img1, fx, fy, cx, cy)
    cv2.imwrite("reprojection/img_pointcloud1.png", img)
    img = project_pointcloud(pointcloud2, img1, fx, fy, cx, cy)
    cv2.imwrite("reprojection/img_pointcloud2.png", img)
    img = project_pointcloud(pointcloud3, img1, fx, fy, cx, cy)
    cv2.imwrite("reprojection/img_pointcloud3.png", img)
    img = project_pointcloud(pointcloud4, img1, fx, fy, cx, cy)
    cv2.imwrite("reprojection/img_pointcloud4.png", img)



dense_stereo_matching(image_pathes, fx, fy, cx, cy, baseline)
