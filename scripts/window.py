import numpy as np
import vispy.scene
from vispy.scene import visuals
import sys
import os
import random
import math
import ransort as rs
frame = input("frame number : ")
print("-"*20)
print("original cloud : o")
print("processed cloud: p")
print("synthesized cloud : s")
print("road extraction : r")
print("road marking extraction : m")
print("-"*20)
print("scene type: ")
type = input()

canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

if type == 'o' :
    root_dir = "data/2011_09_26/2011_09_26_drive_0002_sync/velodyne_points/data/"
    filename = os.listdir(root_dir)
    filename.sort()

    points_array = np.fromfile(root_dir + filename[int(frame)], dtype=np.float32)
    points = np.reshape(points_array, (-1, 4))

elif type == 'p':
    root_dir = "data/2011_09_26/pre-processing/cloud/"
    filename = os.listdir(root_dir)
    filename.sort()

    points = np.asarray(np.load(root_dir + filename[int(frame)])).reshape(-1,4)

elif type == 's':
    root_dir = "data/2011_09_26/combined_road/"
    filename = os.listdir(root_dir)
    filename.sort()

    points = np.asarray(np.load(root_dir + filename[int(frame)-19])).reshape((-1, 4))

elif type == 'r':
    root_dir = "data/2011_09_26/road_extraction/pointcloud/"
    filename = os.listdir(root_dir)
    filename.sort()

    points = np.asarray(np.load(root_dir + filename[int(frame)-19])).reshape(-1,4)

elif type == 'm':
    root_dir = "data/2011_09_26/marking_extraction/"
    filename = os.listdir(root_dir)
    filename.sort()

    points = np.asarray(np.load(root_dir + filename[int(frame)-19])).reshape((-1, 4))

print(len(points))
scatter = visuals.Markers()
intensity_color = []

for i in range(0, len(points)):
    intensity_color.append([1, 1, 1, points[i][3]])

color_array = np.asarray(intensity_color)
scatter.set_data(points[:, :3], edge_color=None, face_color=color_array, size=4)
# calib_points[profile_offsets[0]:profile_offsets[-1], :3] (-30 to 30)
view.add(scatter)

view.camera = 'turntable'

axis = visuals.XYZAxis(parent=view.scene)
vispy.app.run()