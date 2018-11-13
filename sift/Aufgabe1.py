# coding=utf-8
import os
import numpy as np
import cv2
import glob
from Uebungsblatt7.siftdetector import detect_keypoints

image_pathes = glob.glob('images/*.png')
image_pairs = np.array([image_pathes[:2], image_pathes[2:]])

# Methoden zum Anpassen der Rückgabeformate
def to_cv2_kplist(kp): return list(map(to_cv2_kp, kp))


def to_cv2_kp(kp): return cv2.KeyPoint(kp[1], kp[0], kp[2], kp[3] / np.pi * 180)


def to_cv2_di(di): return np.asarray(di, np.float32)


def match_and_draw(img_pairs):
    for pathes in img_pairs:
        image1 = cv2.imread(pathes[0])
        image2 = cv2.imread(pathes[1])

        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        [detected_keypoints1, descriptors1] = detect_keypoints(pathes[0], 5)
        [detected_keypoints2, descriptors2] = detect_keypoints(pathes[1], 5)

        # Rückgabeformat ändern
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
        # Matching durchführen
        matches = bf.match(d1_cv2, d2_cv2)
        print "a. Alle Matches : %d" % len(matches)
        matches_img = cv2.drawMatches(image1, kp1_cv2, image2, kp2_cv2, matches, None)
        cv2.imwrite("kp_images/" + filename1 + "_" + filename2 + "_m.png", matches_img)

        print "b. Die 30 besten Matches"
        sorted_matches = sorted(matches, key=lambda x: x.distance)
        matches_img = cv2.drawMatches(image1, kp1_cv2, image2, kp2_cv2, sorted_matches[:30], None)
        cv2.imwrite("kp_images/" + filename1 + "_" + filename2 + "_m30.png", matches_img)

        print "c. Alle Matches mit einem matching theshold von 0.7 "
        matches = bf.knnMatch(d1_cv2, d2_cv2, k=2)
        good = []
        pts1 = []
        pts2 = []
        theshold_matching = 0.7
        for m, n in matches:
            if m.distance < theshold_matching * n.distance:
                good.append([m])
                pts1.append(kp1_cv2[m.queryIdx].pt)
                pts2.append(kp2_cv2[m.trainIdx].pt)

        good_matches = cv2.drawMatchesKnn(image1, kp1_cv2, image2, kp2_cv2, good, None)
        cv2.imwrite("kp_images/" + filename1 + "_" + filename2 + "_mGood.png", good_matches)

match_and_draw(image_pairs)
