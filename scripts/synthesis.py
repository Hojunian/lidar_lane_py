import numpy as np
import vispy.scene
import sys
import os
import random
import math
import ransort as rs


print('-' * 20)
print('combine processed cloud: p')
print('combine original cloud: o')
print('combine road points: r')
print('-' * 20)
type = input('type : ')

if type =='p':
    velodyne_root_dir = "data/2011_09_26/pre-processing/cloud/"
elif type == 'r':
    velodyne_root_dir = "data/2011_09_26/road_extraction/"
elif type == 'o':
    velodyne_root_dir = "data/2011_09_26/2011_09_26_drive_0002_sync/velodyne_points/data/"

velodyne_filename = os.listdir(velodyne_root_dir)
velodyne_filename.sort()

odometry_root_dir = "data/2011_09_26/2011_09_26_drive_0002_sync/oxts/data/"
odometry_filename = os.listdir(odometry_root_dir)
odometry_filename.sort()

# read point cloud from bin file
velodyne_file = open("data/2011_09_26/2011_09_26_drive_0002_sync/velodyne_points/timestamps.txt", 'r')
velodyne_timestamps = velodyne_file.readlines()
velodyne_file.close()

odometry_file = open("data/2011_09_26/2011_09_26_drive_0002_sync/oxts/timestamps.txt", 'r')
odometry_timestamps = odometry_file.readlines()
odometry_file.close()


def SO3(roll, pitch ,yaw):
    rot_z = np.array([math.cos(yaw), -math.sin(yaw), 0, math.sin(yaw), math.cos(yaw), 0, 0, 0, 1]).reshape(3, 3)
    rot_y = np.array([math.cos(pitch), 0, math.sin(pitch), 0, 1, 0, -math.sin(pitch), 0, math.cos(pitch)]).reshape(3, 3)
    rot_x = np.array([1, 0, 0, 0, math.cos(roll), -math.sin(roll), 0, math.sin(roll), math.cos(roll)]).reshape(3, 3)
    R = np.matmul(rot_z, np.matmul(rot_y, rot_x))
    return R

for frame in range(19, len(velodyne_filename)):

    current_file = open(odometry_root_dir + odometry_filename[frame])
    current_info = current_file.readline().split()
    roll_c = float(current_info[3])
    pitch_c = float(current_info[4])
    yaw_c = float(current_info[5])
    lat_c = float(current_info[0])
    lon_c = float(current_info[1])
    alt_c = float(current_info[2])

    x_c = 6378137 * math.cos(math.pi * lat_c / 180) * math.pi * lon_c / 180
    y_c = 6378137 * math.cos(math.pi * lat_c / 180) * math.log(math.tan(math.pi * (90 + lat_c) / 360))
    z_c = alt_c
    current_file.close()

    R_gc = SO3(roll_c, pitch_c, yaw_c)
    R_cg = np.linalg.inv(R_gc)

    temp_points = []
    for i in range(frame - 19, frame + 1):

        points = np.load(velodyne_root_dir + velodyne_filename[i])

        points_array = np.reshape(points, (-1,4))

        file = open(odometry_root_dir + odometry_filename[i])
        info = file.readline().split()
        roll_i = float(info[3])
        pitch_i = float(info[4])
        yaw_i = float(info[5])
        lat_i = float(info[0])
        lon_i = float(info[1])
        alt_i = float(info[2])
        x_i = 6378137 * math.cos(math.pi * lat_c / 180) * math.pi * lon_i / 180
        y_i = 6378137 * math.cos(math.pi * lat_c / 180) * math.log(math.tan(math.pi * (90 + lat_i) / 360))
        z_i = alt_i
        file.close()

        R_gi = SO3(roll_i, pitch_i, yaw_i)
        v_g = np.array([x_i - x_c, y_i - y_c, z_i - z_c])
        # v : vector from current frame to ith frame
        v_c = np.matmul(R_cg, v_g)
        R_ci = np.matmul(R_cg, R_gi)

        for j in range(0, len(points_array)):
            # remove points whose intensities are too low(weak points) or whose z coordinate is too small(outlier)
            if points_array[j, 3] >= 0.1:
                new_point = np.matmul(R_ci, points_array[j][:3]) + v_c
                temp_points.append(np.array([new_point[0], new_point[1], new_point[2], points_array[j][3]]))


    original_points = np.asarray(temp_points)

    pc_root = "data/2011_09_26/combined_road/" + str(frame) + "(20 combined)"
    np.save(pc_root, original_points)
    print(frame, 'done')


