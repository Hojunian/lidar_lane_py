import numpy as np
import vispy.scene
import sys
import os
import random
import math
import ransort as rs
import cv2
from matplotlib import pyplot as plt
from vispy.scene import visuals
from sklearn.cluster import DBSCAN
import scipy.stats

def get_cov(cluster):
    mean_of_cluster = np.mean(cluster, axis=0)
    cent = np.zeros(cluster.shape)
    cent[:, 0] = cluster[:, 0] - np.ones(cluster.shape[0]) * mean_of_cluster[0]
    cent[:, 1] = cluster[:, 1] - np.ones(cluster.shape[0]) * mean_of_cluster[1]
    cent[:, 2] = cluster[:, 2] - np.ones(cluster.shape[0]) * mean_of_cluster[2]

    cov = (np.transpose(cent) @ cent) / (len(cluster)-1)

    return cov

root_dir = "data/2011_09_26/marking_extraction/"
filename = os.listdir(root_dir)
filename.sort()

frame = 48
points = np.asarray(np.load(root_dir + filename[int(frame)-19])).reshape((-1, 4))
clustering = DBSCAN(eps=0.3, min_samples=5).fit(points)

fig = plt.figure()
ax = fig.gca(projection='3d')


principal_axes = []
covs = []
centers = []
lengths = [-1] * (max(clustering.labels_) + 1)    # 0 if cluster is not valid
for i in range(0, max(clustering.labels_) + 1):
    cluster = points[clustering.labels_[:]==i, :3]


    mean_of_cluster = np.mean(cluster, axis=0)
    cent = np.zeros(cluster.shape)
    cent[:, 0] = cluster[:, 0] - np.ones(cluster.shape[0]) * mean_of_cluster[0]
    cent[:, 1] = cluster[:, 1] - np.ones(cluster.shape[0]) * mean_of_cluster[1]
    cent[:, 2] = cluster[:, 2] - np.ones(cluster.shape[0]) * mean_of_cluster[2]


    U, S, VH = np.linalg.svd(cent)
    centers.append(mean_of_cluster)
    principal_axes.append(VH[0])
    covs.append((np.transpose(cent) @ cent)/(len(cluster)-1))

    if len(cluster) < 7:
        continue
    elif S[0] < 5 * S[1] and len(cluster) > 100:
        continue
    elif VH[0, 2] > 0.1:
        continue
    lengths[i] = len(cluster)

i = 32


print(lengths)
lane_index = []

for i in range(0, 5):
    largest_index = lengths.index(max(lengths))
    print(largest_index)
    while 1:
        count = 0
        largest_cluster = points[clustering.labels_[:] == largest_index, :3]

        largest_cov = get_cov(largest_cluster)
        print(largest_cov)
        largest_mean = np.mean(largest_cluster, axis=0)

        multi_Gaussian = scipy.stats.multivariate_normal(mean=largest_mean, cov=largest_cov)
        for i in range(0, len(points)):

            if clustering.labels_[i] == largest_index or lengths[clustering.labels_[i]] <= 0:
                continue

            if clustering.labels_[i] >= 0 and multi_Gaussian.pdf(points[i, :3]) >= 0.00015:

                lengths[clustering.labels_[i]] -= 1
                clustering.labels_[i] = largest_index
                count += 1


        if count < 5:
            lengths[largest_index] = 0
            lane_index.append(largest_index)
            break

# x = cluster[:, 0]
# y = cluster[:, 1]
# z = cluster[:, 2]
# ax.scatter(x, y, z, s=1, cmap='Pairs')
# ax.view_init(90, 180)