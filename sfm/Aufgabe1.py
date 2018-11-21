# coding=utf-8
import os
import numpy as np
import cv2
import glob
from Uebungsblatt8.siftdetector import detect_keypoints

image_pathes = glob.glob('images/*.png')
image_pairs = np.array([image_pathes[:2], image_pathes[2:]])


def to_cv2_kplist(kp): return list(map(to_cv2_kp, kp))


def to_cv2_kp(kp): return cv2.KeyPoint(kp[1], kp[0], kp[2], kp[3] / np.pi * 180)


def to_cv2_di(di): return np.asarray(di, np.float32)


def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        cv2.line(img1, (x0,y0), (x1,y1), color,1)
        cv2.circle(img1,tuple(pt1),5,color,-1)
        cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def goodmatches(matches, threshold, img1, img2, kp1, kp2, fname1, fname2):
    good = []
    pts1 = []
    pts2 = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append([m])
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    good_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None)
    cv2.imwrite("kp_images/" + fname1 + "_" + fname2 + "_threshold_" + str(threshold) + ".png", good_matches)
    return pts1, pts2


def verify_camerapose(P0,K, R1, R2, t, pts1, pts2):

    P1 = K.dot(np.hstack((R1, t)))
    P2 = K.dot(np.hstack((R1, -t)))
    P3 = K.dot(np.hstack((R2, t)))
    P4 = K.dot(np.hstack((R2, -t)))
    cMs = [P1, P2, P3, P4]
    list = []
    print("verify")
    for p in cMs:
        points3D = cv2.triangulatePoints(P0,p,pts1,pts2)
        list.append(check_3DPoints(points3D))
        print("verify")

    max_value = max(list)
    max_index = list.index(max_value)
    return cMs[max_index]



def check_3DPoints(objectPoints):
    countPositiveDepth = 0
    for o in objectPoints:
        print(o)
        print("check")
        if o[2] > 0:
            countPositiveDepth += 1

    return countPositiveDepth


def match_and_draw(img_pairs):
    for pathes in img_pairs:
        image1 = cv2.imread(pathes[0])
        image2 = cv2.imread(pathes[1])
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        [detected_keypoints1, descriptors1] = detect_keypoints(pathes[0], 5)
        [detected_keypoints2, descriptors2] = detect_keypoints(pathes[1], 5)

        kp1_cv2 = to_cv2_kplist(detected_keypoints1)
        d1_cv2 = to_cv2_di(descriptors1)
        kp2_cv2 = to_cv2_kplist(detected_keypoints2)
        d2_cv2 = to_cv2_di(descriptors2)

        image_out1 = np.array([])
        image_out2 = np.array([])

        # Keypoints einzeichnen
        kp_image1 = cv2.drawKeypoints(gray1, kp1_cv2, image_out1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        kp_image2 = cv2.drawKeypoints(gray2, kp2_cv2, image_out2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        filename_w_ext1 = os.path.basename(pathes[0])
        filename_w_ext2 = os.path.basename(pathes[1])
        filename1 = os.path.splitext(filename_w_ext1)[0]
        filename2 = os.path.splitext(filename_w_ext2)[0]
        # Bilder mit Keypoints abspeichern
        cv2.imwrite("kp_images/" + filename1 + "_kp.png",kp_image1)
        cv2.imwrite("kp_images/" + filename2 + "_kp.png",kp_image2)

        # Zuerst Objekt des Matchers erzeugen
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(d1_cv2, d2_cv2, k=2)

        # Matches mit einem threshold von 0.7
        pts1, pts2 = goodmatches(matches, 0.7, image1, image2, kp1_cv2, kp2_cv2, filename1, filename2)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

        # Punkte die die Epipolareinschränkung einhalten?
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

        # Epipolarlinien berechnen
        lines = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F)
        lines = lines.reshape(-1, 3)
        # Epipolarlinien in das Rechte bild einzeichnen
        img1, img2 = drawlines(gray2, gray1, lines, pts2, pts1)
        cv2.imwrite("kp_images/" + filename1 + "_" + filename2 + "_epilines1.png", img1)
        cv2.imwrite("kp_images/" + filename1 + "_" + filename2 + "_epilines2.png", img2)

        # ------------------------------------------------------------------------------------------------------------
        matches = bf.knnMatch(d1_cv2, d2_cv2, k=2)
        # Matches mit einem threshold von 0.8
        pts1, pts2 = goodmatches(matches, 0.8, image1, image2, kp1_cv2, kp2_cv2, filename1, filename2)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
        # Punkte die die Epipolareinschränkung einhalten
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

        lines = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F)
        lines = lines.reshape(-1, 3)
        img5, img6 = drawlines(gray2, gray1, lines, pts2, pts1)
        cv2.imwrite("kp_images/" + filename1 + "_" + filename2 + "_epilines5.png", img5)
        cv2.imwrite("kp_images/" + filename1 + "_" + filename2 + "_epilines6.png", img6)
        """ print(F)
                print(pts1)
                print(pts2)
                # Aufgabe 2
                fx = fy = 721.5
                cx = 690.5
                cy = 172.8
                print("-----")
                K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
                Rt = np.hstack((np.eye(3), np.zeros((3, 1))))
                P0 = K.dot(Rt)
                print("-----")
                E = K.T * np.mat(F) * K
                R1, R2, t = cv2.decomposeEssentialMat(E)
                print("-----")
                P1 = verify_camerapose(P0, K, R1, R2, t, pts1.reshape(-1, 1, 2), pts2.reshape(-1, 1, 2))
                print("-----")
                pointcloud = cv2.triangulatePoints(P0, P1, pts1, pts2)
                write_ply(filename1 + 'punktwolke.ply', pointcloud)"""



match_and_draw(image_pairs)


