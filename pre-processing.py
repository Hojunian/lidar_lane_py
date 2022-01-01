import numpy as np
import os
import ransort as rs
import math

velodyne_root_dir = "data/2011_09_26/2011_09_26_drive_0002_sync/velodyne_points/data/"
velodyne_filename = os.listdir(velodyne_root_dir)
velodyne_filename.sort()



count = 0
for file in velodyne_filename:
    points = np.fromfile(velodyne_root_dir + file, dtype=np.float32)
    original_points = np.reshape(points, (-1,4))

    for point in original_points:
        r = math.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)
        point[3] = min(point[3] * (0.8 * math.log10(r) + 6 / (r ** 2)), 1)

    total_num = len(original_points)
    inside_num = 0
    C = [0] * 240
    for i in range(0, total_num):
        if -30 <= original_points[i, 0] < 30:
            index = int((original_points[i, 0] + 30) * 4 // 1)
            C[index] = C[index] + 1
            inside_num += 1

    B = [[]]*inside_num

    for i in range(1,240):
        C[i] = C[i] + C[i-1]

    for i in range(0, total_num):
        if -30 <= original_points[total_num-1-i, 0] < 30:
            x = int((original_points[total_num-1-i, 0] + 30) * 4 // 1)
            B[C[x]-1] = original_points[total_num-1-i]
            C[x] = C[x] - 1

    B.append([35, 1, 1, 0])
    calib_points = np.asarray(B)


    # divide point cloud by profiles -30m to 30m, width 0.25m, total : 240 profiles
    profile_offsets = [0]*241    # start index of each profile(divide by x)
    profile_num = 0

    for i in range(0,len(calib_points)):
        if profile_num >= 241:
            break
        elif calib_points[i, 0] >= -30 + profile_num * 0.25:
            profile_offsets[profile_num] = i
            profile_num += 1

    for i in range(0,240):
        rs.randomized_quicksort(calib_points, profile_offsets[i], profile_offsets[i+1], 2)    # sort each profile by z

    first_removal_offsets = [0]*240           # 1+ index of the highest point in the lowest layer
    for i in range(0,240):
        first_removal_offsets[i] = profile_offsets[i+1]
        for j in range(profile_offsets[i]+1, profile_offsets[i+1]):
            diff = calib_points[j, 2]-calib_points[j-1, 2]
            if diff >= 0.1 and j - profile_offsets[i] >= 10:
                first_removal_offsets[i] = j
                break

    saver = []
    offsets = [0]*241
    count = 0
    for i in range(0, 240):
        offsets[i] = count
        for j in range(profile_offsets[i], first_removal_offsets[i]):
            if -20 <= calib_points[j, 1] <= 20:
                saver.append(calib_points[j])
                count += 1

    offsets[240] = len(saver)

    pc_root = "data/2011_09_26/pre-processing/cloud/ " + file
    offsets_root = "data/2011_09_26/pre-processing/offsets/ " + file
    np.save(pc_root, np.asarray(saver))
    np.save(offsets_root, np.asarray(offsets))

    print(file,"done")
    count += 1

