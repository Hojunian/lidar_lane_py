import numpy as np
import vispy.scene
from vispy.scene import visuals
import sys
import os
import random
import math
import ransort as rs
from matplotlib import pyplot as plt

frame = 40

root_dir = "data/2011_09_26/2011_09_26_drive_0002_sync/velodyne_points/data/"
filename = os.listdir(root_dir)
filename.sort()

points_array = np.fromfile(root_dir + filename[int(frame)], dtype=np.float32)
points = np.reshape(points_array, (-1, 4))

# root_dir = "data/2011_09_26/road_extraction/pointcloud/"
# filename = os.listdir(root_dir)
# filename.sort()
#
# points = np.asarray(np.load(root_dir + filename[int(frame)-19])).reshape(-1,4)

distance = []
intensities = []
for point in points:
    r = math.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
    distance.append(r)
    intensities.append(min(point[3] * (0.8 * math.log10(r) + 6/(r**2)), 1))

print(min(distance))
plt.figure()
plt.scatter(distance, intensities, s=1)
plt.show()

#
# canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
# view = canvas.central_widget.add_view()
#
# scatter = visuals.Markers()
# intensity_color = []
#
# for i in range(80000, 90000):
#     r = math.sqrt(points[i][0] ** 2 + points[i][1] ** 2 + points[i][2] ** 2)
#     calib_intensity = min(points[i][3] * (math.log10(r) + 7/(r**2)), 1)
#     intensity_color.append([1, 1, 1, calib_intensity])
#
# color_array = np.asarray(intensity_color)
# scatter.set_data(points[80000:90000, :3], edge_color=None, face_color=color_array, size=2.5)
# # calib_points[profile_offsets[0]:profile_offsets[-1], :3] (-30 to 30)
# view.add(scatter)
#
# view.camera = 'turntable'
#
# axis = visuals.XYZAxis(parent=view.scene)
# vispy.app.run()