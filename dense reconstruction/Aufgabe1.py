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
        np.savetxt(f, verts, '%f %f %f %d %d %d' )


def get_depth_img(f, tx, d):
    depth = np.zeros(d.shape)
    rows = d.shape[0]
    cols = d.shape[1]
    for x in range(0, rows):
        for y in range(0, cols):
            if d[x,y] == 0:
                depth[x,y] = 0
            else:
                depth[x,y] = (f*tx)/d[x,y]
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
                vector.append([X, Y, Z, img1[x, y][0], img1[x, y][1], img1[x, y][2]])
    return vector


def dense_stereo_matching(img_pathes, fx, fy, cx, cy, baseline):
    print("Aufgabe1: Disparität")
    blocksizes = [4, 8, 12, 16] # Gibt Fenstergröße an (1-20)
    min_disp = 1 # Gibt minimale Disparität an (0-10)
    y = 5
    num_disp = 16 * y # Gibt maximale Disparität an (16 * y)
    img1 = cv2.imread(img_pathes[0], 1)
    img2 = cv2.imread(img_pathes[1], 1)

    for blocksize in blocksizes:
        # Aufgabe 1
        stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=blocksize)
        disparity = stereo.compute(img1, img2).astype(np.float32)/16.0
        depth = get_depth_img(fx, baseline, disparity)

        disparity = cv2.normalize(disparity, np.zeros(disparity.shape), 0, 255, cv2.NORM_MINMAX)
        img_out = cv2.applyColorMap(disparity.astype(np.uint8),cv2.COLORMAP_JET)
        cv2.imwrite("disparity_images/" + "disp_b"+ str(blocksize) + ".png", img_out)

        # Aufgabe 2.1
        depth_norm = cv2.normalize(depth, np.zeros(depth.shape), 0, 255, cv2.NORM_MINMAX)
        img_out = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite("depth_images/" + "depth_b" + str(blocksize) + ".png", img_out)

        # Aufgabe 2.2
        vector = get_pointcloud(img1, depth)
        write_plyC("pointclouds/pointscloud" + str(blocksize) + ".ply", np.array(vector))

        # Aufgabe 3
        stereo = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=blocksize,
                                       speckleWindowSize=100, speckleRange=1)
        disparity = stereo.compute(img1, img2).astype(np.float32) / 16.0
        depth = get_depth_img(fx, baseline, disparity)

        disparity = cv2.normalize(disparity, np.zeros(disparity.shape), 0, 255, cv2.NORM_MINMAX)
        img_out = cv2.applyColorMap(disparity.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite("disparity_images/" + "disp_noise_reduced_b" + str(blocksize) + ".png", img_out)

        depth_norm = cv2.normalize(depth, np.zeros(depth.shape), 0, 255, cv2.NORM_MINMAX)
        img_out = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite("depth_images/" + "depth_noise_reduced_b" + str(blocksize) + ".png", img_out)

        vector = get_pointcloud(img1, depth)
        write_plyC("pointclouds/pointscloud_noise_reduced" + str(blocksize) + ".ply", np.array(vector))






dense_stereo_matching(image_pathes, fx, fy, cx, cy, baseline)

