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


frame = 70
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

calib_points = np.asarray(np.load("data/2011_09_26/multi-frame/synthesis_points: " + str(frame-9) + "to" + str(frame) + ".npy")).reshape((-1, 4))
profile_offsets = np.load("data/2011_09_26/multi-frame/offsets: " + str(frame-9) + "to" + str(frame) + ".npy")
first_removal_offsets = profile_offsets[1:]

# Detecting curb from the data
curb_offsets = [[0,0]]*240
y_profile_offsets = [[]]*240
for i in range(0,240):
    rs.randomized_quicksort(calib_points, profile_offsets[i], first_removal_offsets[i], 1)   # sort 1 time removed prof by y
    y_profile_offsets[i] = [profile_offsets[i]]
    first_y_point_offset = profile_offsets[i]

    for j in range(profile_offsets[i], first_removal_offsets[i]):
        if calib_points[j, 1] - calib_points[first_y_point_offset, 1] >= 0.15:
            y_profile_offsets[i].append(j)
            first_y_point_offset = j

    y_profile_offsets[i].append(first_removal_offsets[i])
    curb_estimated_offsets = 0
    first_road_offset = 0
    curb_offsets[i] = [first_road_offset, len(y_profile_offsets[i]) - 1]
    for j in range(0, len(y_profile_offsets[i])-1):
        height_diff = max(calib_points[y_profile_offsets[i][j]:y_profile_offsets[i][j+1], 2]) - min(calib_points[y_profile_offsets[i][j]:y_profile_offsets[i][j+1], 2])
        if height_diff >= 0.10:
            if calib_points[y_profile_offsets[i][curb_estimated_offsets], 1] * calib_points[y_profile_offsets[i][j], 1] >=0:
                curb_estimated_offsets = j
                first_road_offset = j+1
                curb_offsets[i] = [first_road_offset, len(y_profile_offsets[i]) - 1]
            else:
                curb_offsets[i] = [first_road_offset, j]
                break


# Otsu thresholding
intensity_threshold = [[]]*240
for i in range(0, 240):
    intensity_threshold[i] = [0] * curb_offsets[i][1]
    for k in range(curb_offsets[i][0], curb_offsets[i][1]):
        max_marking_var = 0
        for j in range(0, 100):
            marking_var = 0
            road_intensity = []
            marking_intensity = []
            for l in range(y_profile_offsets[i][k], y_profile_offsets[i][k+1]):
                if calib_points[l][3] >= j/100:
                    marking_intensity.append(calib_points[l][3])
                else:
                    road_intensity.append(calib_points[l][3])
            if len(road_intensity) == 0 or len(marking_intensity) == 0:
                continue
            road_intensity_mean = sum(road_intensity) / len(road_intensity)
            marking_intensity_mean = sum(marking_intensity) / len(marking_intensity)
            road_prob = len(road_intensity) / (y_profile_offsets[i][k+1] - y_profile_offsets[i][k])
            marking_prob = len(marking_intensity) / (y_profile_offsets[i][k+1] - y_profile_offsets[i][k])
            marking_var = road_prob * marking_prob * (road_intensity_mean - marking_intensity_mean) ** 2
            if marking_var > max_marking_var:
                max_marking_var = marking_var
                intensity_threshold[i][k] = j/100





# collect road points to visualize
raw_road = []
for i in range(0,240):
    for j in range(curb_offsets[i][0], curb_offsets[i][1]):
        if calib_points[y_profile_offsets[i][j]][3] >= 0:          #intensity_threshold[i][j]
            raw_road.append(calib_points[y_profile_offsets[i][j]])
road_points = np.asarray(raw_road)
np.save("data/2011_09_26/road_extraction/70", road_points)
print(len(road_points))
# to visualize intensity by color
scatter = visuals.Markers()
intensity_color = []
max_intensity = max(road_points[:, 3])
min_intensity = min(road_points[:, 3])

for i in range(0,len(road_points)):             # profile_offsets[0],profile_offsets[-1] (-30~30) 0,len(calib_points)
    intensity_color.append([1, 1, 1, road_points[i][3]])
    # (road_points[i,3]-min_intensity)/(max_intensity-min_intensity)

color_array = np.asarray(intensity_color)
scatter.set_data(road_points[:, :3], edge_color=None, face_color=color_array, size=2.5)
# calib_points[profile_offsets[0]:profile_offsets[-1], :3] (-30 to 30)
view.add(scatter)

view.camera = 'turntable'

axis = visuals.XYZAxis(parent=view.scene)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if sys.flags.interactive !=1:
        vispy.app.run()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
