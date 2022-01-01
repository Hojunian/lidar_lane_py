import numpy as np
import vispy.scene
import sys
import os
import random
import math
import ransort as rs
import cv2
from matplotlib import pyplot as plt

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

cloud_root_dir = "data/2011_09_26/road_extraction/pointcloud/"
cloud_filename = os.listdir(cloud_root_dir)
cloud_filename.sort()

intensity_root_dir = "data/2011_09_26/road_extraction/medians/"
intensity_filename = os.listdir(intensity_root_dir)
intensity_filename.sort()

roadpix_root_dir = "data/2011_09_26/road_extraction/road_pixel/"
roadpix_filename = os.listdir(roadpix_root_dir)
roadpix_filename.sort()

offsets_root_dir = "data/2011_09_26/road_extraction/offsets_for_pixel/"
offsets_filename = os.listdir(offsets_root_dir)
offsets_filename.sort()


for frame in range(19, len(cloud_filename)+19):

    markings = []

    points = np.asarray(np.load(cloud_root_dir + cloud_filename[int(frame)-19])).reshape((-1, 4))
    road_pixels = np.load(roadpix_root_dir + roadpix_filename[int(frame)-19])
    intensity_medians = np.load(intensity_root_dir + intensity_filename[int(frame)-19])
    offsets_for_each_pixel = np.load(offsets_root_dir + offsets_filename[int(frame)-19])

    # Otsu thresholding for entire road points
    int_arr = []
    for point in points:
        int_arr.append(point[3])

    int_image = (np.asarray(int_arr) * 255).astype(np.uint8)
    first_threshold, tt = cv2.threshold(int_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # second Otsu thresholding for each area (5m*5m)
    threshold = np.zeros((14, 8))
    for x in range(0, 14):
        for y in range(0, 8):
            intensities = []

            for i in range(50 * x, 50 * x + 50):
                for j in range(50 * y, 50 * y + 50):
                    for offset in offsets_for_each_pixel[i, j]:
                        if points[offset][3] >= first_threshold / 255:
                            intensities.append(points[offset][3])

            threshold[x, y], sth = cv2.threshold((np.asarray(intensities) * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # for i in range(50 * x, 50 * x + 50):
            #     for j in range(50 * y, 50 * y + 50):
            #         count = 0
            #         for offset in offsets_for_each_pixel[i, j]:
            #             if points[offset][3] >= threshold[x, y] / 255:
            #                 count += 1
            #                 if count >=3:
            #                     markings.append(points[offset])

    # third Otsu thresholding for each area (10m*10m)
    first_markings = []
    final_int = []
    for x in range(0, 7):
        for y in range(0, 4):
            intensities = []

            for i in range(100 * x, 100 * x + 100):
                for j in range(100 * y, 100 * y + 100):
                    for offset in offsets_for_each_pixel[i, j]:
                        if points[offset][3] >= threshold[i//50, j//50] / 255:
                            intensities.append(points[offset][3])

            ths, sth = cv2.threshold((np.asarray(intensities) * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            for i in range(100 * x, 100 * x + 100):
                for j in range(100 * y, 100 * y + 100):
                    count = 0
                    for offset in offsets_for_each_pixel[i, j]:
                        if points[offset][3] >= ths / 255:
                            count += 1
                            if count >= 2:
                                markings.append(points[offset])


                            # first_markings.append(points[offset])
                            # final_int.append(points[offset][3])

    # final_threshold, ttf = cv2.threshold((np.asarray(final_int) * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # for pp in first_markings:
    #     if pp[3] >= final_threshold/255:
    #         markings.append(pp)



    # indicator = np.zeros((700, 400))
    # for pixel in road_pixels:
    #     indicator[pixel[0], pixel[1]] = 1
    #
    # intensity_medians = np.asarray(intensity_medians) * indicator
    # img = (intensity_medians*255).astype(np.uint8)
    # img = cv2.medianBlur(img, 5)
    # ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # plt.figure()
    # plt.imshow(img, 'gray')
    # plt.axis('off')
    # # plt.savefig('data/2011_09_26/marking_image/' + str(frame) + '-hist.jpeg')
    # plt.show()
    # # plt.close()



    # for x in range(0, 70):
    #     for y in range(0, 40):
    #
    #         intensity_threshold = 0
    #         max_marking_var = 0
    #
    #         for th in range(0,100):
    #
    #             marking_var = 0
    #             road_intensity = []
    #             marking_intensity = []
    #
    #             for i in range(10*x, 10*x + 10):
    #                 for j in range(10*y, 10*y + 10):
    #
    #                     if intensity_medians[i, j] >= th / 100 and intensity_medians[i, j] != 0:
    #                         marking_intensity.append(intensity_medians[i, j])
    #                     elif intensity_medians[i, j] != 0:
    #                         road_intensity.append(intensity_medians[i, j])
    #
    #             if len(road_intensity) == 0 or len(marking_intensity) == 0:
    #                 continue
    #
    #             road_intensity_mean = sum(road_intensity) / len(road_intensity)
    #             marking_intensity_mean = sum(marking_intensity) / len(marking_intensity)
    #             road_prob = len(road_intensity) / (len(road_intensity) + len(marking_intensity))
    #             marking_prob = len(marking_intensity) / (len(road_intensity) + len(marking_intensity))
    #             marking_var = road_prob * marking_prob * (road_intensity_mean - marking_intensity_mean) ** 2
    #
    #             if marking_var > max_marking_var:
    #                 max_marking_var = marking_var
    #                 intensity_threshold = th / 100
    #
    #         for i in range(10 * x, 10 * x + 10):
    #             for j in range(10 * y, 10 * y + 10):
    #                 if intensity_medians[i, j] < intensity_threshold:
    #                     intensity_medians[i, j] = 0
    #

    markings_array = np.asarray(markings)

    np.save("data/2011_09_26/marking_extraction/" + str(frame), markings_array)
    print(frame, "done")