# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import vispy.scene
from vispy.scene import visuals
import sys
import os
import random
import math
import ransort as rs

cloud_root_dir = "data/2011_09_26/combined_road/"
pc_files = os.listdir(cloud_root_dir)
pc_files.sort()

for i in range(19, len(pc_files) + 19):

    calib_points = np.asarray(np.load(cloud_root_dir + pc_files[i-19])).reshape((-1, 4))

    # set 10cm * 10cm pixels inside -40<=x<30, -20<=y<20
    z_inside_pixel = []
    for p in range(0,700):
        line = []
        for q in range(0, 400):
            line.append([-999])  # -999 : empty pixel
        z_inside_pixel.append(line)

    offsets_inside_pixel = []
    for p in range(0, 700):
        line = []
        for q in range(0, 400):
            line.append([])
        offsets_inside_pixel.append(line)

    intensity_inside_pixel = []
    for p in range(0,700):
        line = []
        for q in range(0, 400):
            line.append([-999])
        intensity_inside_pixel.append(line)

    for p in range(0, len(calib_points)):
        if -40 <= calib_points[p, 0] < 30 and -20 <= calib_points[p, 1] < 20:
            x_pixel = int((calib_points[p, 0] // 0.1) + 400)
            y_pixel = int((calib_points[p, 1] // 0.1) + 200)

            z_inside_pixel[x_pixel][y_pixel].append(calib_points[p, 2])
            intensity_inside_pixel[x_pixel][y_pixel].append(calib_points[p, 3])
            offsets_inside_pixel[x_pixel][y_pixel].append(p)

    z_medians = []
    for p in range(0, 700):
        line = []
        for q in range(0, 400):
            line.append(-999)
        z_medians.append(line)

    intensity_medians = []
    for p in range(0, 700):
        line = []
        for q in range(0, 400):
            line.append(-999)
        intensity_medians.append(line)

    for x in range(0, 700):
        for y in range(0, 400):
            if len(z_inside_pixel[x][y]) > 1:
                z_inside_pixel[x][y].sort()
                z_medians[x][y] = z_inside_pixel[x][y][len(z_inside_pixel[x][y])//2]

                intensity_inside_pixel[x][y].sort()
                intensity_medians[x][y] = intensity_inside_pixel[x][y][len(intensity_inside_pixel[x][y])//2]


    sure_x = 400
    sure_y = 200
    breaker = False

    for x in range(360, 440):
        for y in range(160, 240):

            if -1.75 < z_medians[x][y] < -1.71 and -1.75 < z_medians[x-1][y] < -1.71 and -1.75 < z_medians[x][y-1] < -1.71:
                z_inside_pixel[x][y][0] = 999
                sure_x = x
                sure_y = y
                breaker = True
                break
        if breaker:
            break


    road_pixels = []
    road_pixels.append([sure_x, sure_y])

    while True:
        checker = 0
        for pixel in road_pixels:
            diff = []
            if 1 <= pixel[0] <= 698 and 1 <= pixel[1] <= 398:

                if abs(z_medians[pixel[0]][pixel[1]] - z_medians[pixel[0] - 1][pixel[1]]) < 0.025 and z_inside_pixel[pixel[0] - 1][pixel[1]][0] == -999:
                    road_pixels.append([pixel[0] - 1, pixel[1]])
                    z_inside_pixel[pixel[0] - 1][pixel[1]][0] = 999
                    checker += 1

                if abs(z_medians[pixel[0]][pixel[1]] - z_medians[pixel[0] + 1][pixel[1]]) < 0.025 and z_inside_pixel[pixel[0] + 1][pixel[1]][0] == -999:
                    road_pixels.append([pixel[0] + 1, pixel[1]])
                    z_inside_pixel[pixel[0] + 1][pixel[1]][0] = 999
                    checker += 1

                if abs(z_medians[pixel[0]][pixel[1]] - z_medians[pixel[0]][pixel[1] - 1]) < 0.025 and z_inside_pixel[pixel[0]][pixel[1] - 1][0] == -999:
                    road_pixels.append([pixel[0], pixel[1] - 1])
                    z_inside_pixel[pixel[0]][pixel[1] - 1][0] = 999
                    checker += 1

                if abs(z_medians[pixel[0]][pixel[1]] - z_medians[pixel[0]][pixel[1] + 1]) < 0.025 and z_inside_pixel[pixel[0]][pixel[1] + 1][0] == -999:
                    road_pixels.append([pixel[0], pixel[1] + 1])
                    z_inside_pixel[pixel[0]][pixel[1] + 1][0] = 999
                    checker += 1

        if checker == 0:
            break

    raw_road = []
    offsets_saver = []
    for p in range(0, 700):
        line = []
        for q in range(0, 400):
            line.append([])
        offsets_saver.append(line)
    raw_offset = 0
    for p in road_pixels:
        med = z_medians[p[0]][p[1]]
        for offset in offsets_inside_pixel[p[0]][p[1]]:
            if abs(calib_points[offset, 2] - med) < 0.025:
                raw_road.append(calib_points[offset])
                offsets_saver[p[0]][p[1]].append(raw_offset)
                raw_offset += 1

    road_points = np.asarray(raw_road)
    # calibrate intensity with distance

    np.save("data/2011_09_26/road_extraction/pointcloud/" + str(i), road_points)
    np.save("data/2011_09_26/road_extraction/medians/" + str(i), np.asarray(intensity_medians))
    np.save("data/2011_09_26/road_extraction/offsets_for_pixel/" + str(i), np.asarray(offsets_saver))
    np.save("data/2011_09_26/road_extraction/road_pixel/" + str(i), np.asarray(road_pixels))
    print(i, "done")
