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

def get_svd(cluster):
    mean_of_cluster = np.mean(cluster, axis=0)
    cent = np.zeros(cluster.shape)
    cent[:, 0] = cluster[:, 0] - np.ones(cluster.shape[0]) * mean_of_cluster[0]
    cent[:, 1] = cluster[:, 1] - np.ones(cluster.shape[0]) * mean_of_cluster[1]
    cent[:, 2] = cluster[:, 2] - np.ones(cluster.shape[0]) * mean_of_cluster[2]

    U, S, VH = np.linalg.svd(cent)

    return U, S, VH

root_dir = "data/2011_09_26/marking_extraction/"
filename = os.listdir(root_dir)
filename.sort()

for frame in range(41, len(filename)+19):
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

    lane_index = []

    for i in range(0, 5):
        largest_index = lengths.index(max(lengths))
        while 1:
            count = 0
            largest_cluster = points[clustering.labels_[:] == largest_index, :3]

            largest_cov = get_cov(largest_cluster)
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
                U, S, VH = get_svd(largest_cluster)

                if S[0] > 10 * S[1] and len(largest_cluster) > 500:
                    lane_index.append(largest_index)

                lengths[largest_index] = 0

                break

    lane_count = 1
    for i in lane_index:
        cluster = points[clustering.labels_[:] == i, :3]
        np.save("data/2011_09_26/lane_points/" + str(frame) + "_" + str(lane_count), cluster)
        x = cluster[:, 0]
        y = cluster[:, 1]
        z = cluster[:, 2]
        ax.scatter(x, y, z, s=1, cmap='blue')
        ax.view_init(90, 180)
        lane_count += 1

    no_lane_points = []

    for i in range(0, len(points)):
        if lengths[clustering.labels_[i]] < 0 or clustering.labels_[i] in lane_index or clustering.labels_[i] < 0 or lengths[clustering.labels_[i]] > 150:
            continue
        no_lane_points.append(points[i])

    no_lane_points = np.asarray(no_lane_points)
    second_clustering = DBSCAN(eps=0.5, min_samples=5).fit(no_lane_points)
    second_p_axes = []
    second_mean = []
    second_cluster_index = []
    second_cluster_length = []

    for i in range(0, max(second_clustering.labels_) + 1):

        cluster = no_lane_points[second_clustering.labels_[:] == i, :3]
        if len(cluster) > 300:
            continue

        U, S, VH = get_svd(cluster)

        if len(cluster) < 7 or len(cluster) > 150:
            continue
        elif S[0] < 3 * S[1] and len(cluster) > 50:
            continue
        elif VH[0, 2] > 0.1 or VH[0, 0] < 0.95:
            continue
        second_cluster_index.append(i)
        second_mean.append(np.mean(cluster, axis=0))
        second_p_axes.append(VH[0])
        second_cluster_length.append(len(cluster))

    dot_candidate = []
    inside_loc = [-1]*len(second_cluster_index)

    for i in range(0, len(second_cluster_index) - 1):
        if inside_loc[i] == -1:
            dot_candidate.append([i])
            inside_loc[i] = len(dot_candidate) - 1
        for j in range(i+1, len(second_cluster_index)):
            if inside_loc[j] > 0:
                continue
            p_axes_1 = second_p_axes[i]
            p_axes_2 = second_p_axes[j]
            center_vector = second_mean[j] - second_mean[i]
            if np.linalg.norm(center_vector) > 7.5 or abs(center_vector[1]) > 1:
                continue
            center_vector = center_vector / np.linalg.norm(center_vector)

            if abs(np.dot(p_axes_1, p_axes_2)) > 0.98 and abs(np.dot(center_vector, p_axes_1)) > 0.98 and abs(np.dot(center_vector, p_axes_2)) > 0.98:
                inside_loc[j] = inside_loc[i]
                dot_candidate[inside_loc[i]].append(j)
    dot_count = 1
    for group in dot_candidate:
        if len(group) == 1:
            continue
        real_indexes = []
        for j in group:
            real_indexes.append(second_cluster_index[j])
        cluster = []

        for i in range(0, len(no_lane_points)):
            if second_clustering.labels_[i] in real_indexes:
                cluster.append(no_lane_points[i, :3])

        cluster = np.asarray(cluster)

        U, S, VH = get_svd(cluster)
        if S[0] > S[1] * 20 and len(cluster) > 50:
            x = cluster[:, 0]
            y = cluster[:, 1]
            z = cluster[:, 2]
            ax.scatter(x, y, z, s=1, cmap='green')
            ax.view_init(90, 180)
            np.save("data/2011_09_26/dot_lane_points/" + str(frame) + "-" + str(dot_count), cluster)
            dot_count += 1
    # plt.xlim(-30,15)
    # plt.ylim(-5, 12.5)
    # plt.show()
    # plt.savefig('data/2011_09_26/dot_lane/' + str(frame) + '.png')




    # for i in dot_lane_index:
    #     cluster = points[clustering.labels_[:] == i, :3]
    #     x = cluster[:, 0]
    #     y = cluster[:, 1]
    #     z = cluster[:, 2]
    #     ax.scatter(x, y, z, s=1, cmap='Pairs')
    #     ax.view_init(90, 180)

    # plt.savefig('data/2011_09_26/cluster_5kernel/' + str(frame) + '.png')

    # x = cluster[:, 0]
    # y = cluster[:, 1]
    # z = cluster[:, 2]
    # ax.scatter(x, y, z, s=1, cmap='Pairs')
    # ax.view_init(90, 180)

    # plt.savefig('data/2011_09_26/cluster_img/' + str(frame) + '.png')
    #
    # new_points = []
    # for i in range(0, len(points)):
    #     if valid[clustering.labels_[i]]:
    #         new_points.append(points[i])
    #
    # new_points = np.asarray(new_points)
    # print(new_points.shape)
    # clustering = DBSCAN(eps=0.3, min_samples=2).fit(new_points)
    #
    # for i in range(0, max(clustering.labels_) + 1):
    #
    #     cluster = new_points[clustering.labels_[:]==i, :3]
    #     ean_of_cluster = np.mean(cluster, axis=0)
    #     cent = np.zeros(cluster.shape)
    #     cent[:, 0] = cluster[:, 0] - np.ones(cluster.shape[0]) * mean_of_cluster[0]
    #     cent[:, 1] = cluster[:, 1] - np.ones(cluster.shape[0]) * mean_of_cluster[1]
    #     cent[:, 2] = cluster[:, 2] - np.ones(cluster.shape[0]) * mean_of_cluster[2]
    #
    #     U, S, VH = np.linalg.svd(cent)
    #     if len(cluster) < 20:
    #         continue
    #     elif S[0] < 3 * S[1]:
    #         continue
    #     x = cluster[:, 0]
    #     y = cluster[:, 1]
    #     z = cluster[:, 2]
    #     ax.scatter(x, y, z, s=1, cmap='Pairs')
    #     ax.view_init(90, 180)
    # plt.show()


    # large_clusters = []
    # belongs = [-1] * len(centers)
    # for i in range(0, max(clustering.labels_) + 1):
    #     if i == 0:
    #         large_clusters.append([i])
    #         belongs[0] = 0
    #     else:
    #         for j in range(0, i):
    #             if len(points[clustering.labels_[:] == j,:3]) > 10:
    #                 center_to_center = centers[i] - centers[j]
    #                 center_to_center /= np.linalg.norm(center_to_center)
    #
    #                 if np.sum(principal_axes[i] * principal_axes[j]) > 0.95 and np.sum(center_to_center * principal_axes[j]) > 0.95:
    #                     large_clusters[belongs[j]].append(i)
    #                     belongs[i] = belongs[j]
    #                     break
    #         if belongs[i] == -1:
    #             large_clusters.append([i])
    #             belongs[i] = len(large_clusters) - 1
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # for big in large_clusters:
    #     valid = [False] * len(points)
    #     for i in range(0, len(points)):
    #         if clustering.labels_[i] in big:
    #             valid[i] = True
    #
    #     cluster = points[valid, :3]
    #     # if len(cluster) < 100:
    #     #     continue
    #
    #     x = cluster[:, 0]
    #     y = cluster[:, 1]
    #     z = cluster[:, 2]
    #     ax.scatter(x, y, z, s=1, cmap='Pairs')
    #     ax.view_init(90, 180)
    # plt.show()



## TODO : 뭉텡이 버리고 principal axes 기준으로 cluster 합치기

    # for i in range(0, max(clustering.labels_) + 1):
    #
    #     cluster = points[clustering.labels_[:]==i, :3]
    #
    #     # PCA
    #     mean_of_cluster = np.mean(cluster, axis=0)
    #     cent = np.zeros(cluster.shape)
    #     cent[:, 0] = cluster[:, 0] - np.ones(cluster.shape[0]) * mean_of_cluster[0]
    #     cent[:, 1] = cluster[:, 1] - np.ones(cluster.shape[0]) * mean_of_cluster[1]
    #     cent[:, 2] = cluster[:, 2] - np.ones(cluster.shape[0]) * mean_of_cluster[2]
    #
    #
    #     w, v = np.linalg.eig(np.matmul(np.transpose(cent), cent))
    #     sorted_eigval = np.sort(w)
    #
    #     cluster_centers.append(mean_of_cluster)
    #     cluster_directions.append(v[w[:] == sorted_eigval[2]].reshape(3))
    #
    #
    # large_clusters = []
    # belongs = [-1] * len(cluster_centers)
    # for i in range(0, max(clustering.labels_) + 1):
    #     if i == 0:
    #         large_clusters.append([i])
    #         belongs[0] = 0
    #     else:
    #         for j in range(0, i):
    #             if len(points[clustering.labels_[:] == j,:3]) > 10:
    #                 center_to_center = cluster_centers[i] - cluster_centers[j]
    #                 center_to_center /= np.linalg.norm(center_to_center)
    #
    #                 if np.sum(cluster_directions[i] * cluster_directions[j]) > 0.95 and np.sum(center_to_center * cluster_directions[j]) > 0.95:
    #                     large_clusters[belongs[j]].append(i)
    #                     belongs[i] = belongs[j]
    #                     break
    #         if belongs[i] == -1:
    #             large_clusters.append([i])
    #             belongs[i] = len(large_clusters) - 1
    #
    # for big in large_clusters:
    #     valid = [False] * len(points)
    #     for i in range(0, len(points)):
    #         if clustering.labels_[i] in big:
    #             valid[i] = True
    #
    #     cluster = points[valid, :3]
    #     if len(cluster) < 100:
    #         continue
    #     x = cluster[:, 0]
    #     y = cluster[:, 1]
    #     z = cluster[:, 2]
    #     ax.scatter(x, y, z, s=1, cmap='Pairs')
    #     ax.view_init(90, 180)
    #
    # plt.show()





        # if sorted_eigval[2] > 5 * sorted_eigval[1]:
        #     x = cluster[:, 0]
        #     y = cluster[:, 1]
        #     z = cluster[:, 2]
        #     ax.scatter(x, y, z, s=1, cmap='Pairs')
        #     ax.view_init(90, 180)

    # plt.savefig('data/2011_09_26/cluster_img/' + str(frame) + '.png')
    # plt.show()


    # canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    # view = canvas.central_widget.add_view()
    #
    # scatter = visuals.Markers()
    # intensity_color = []
    #
    # for i in range(0, len(points)):
    #     if clustering.labels_[i] != -1:
    #         intensity_color.append([1, 1, 1, points[i][3]])
    #     else:
    #         intensity_color.append([1, 1, 1, 0.01])
    #
    #
    # color_array = np.asarray(intensity_color)
    # scatter.set_data(points[:, :3], edge_color=None, face_color=color_array, size=2.5)
    # # calib_points[profile_offsets[0]:profile_offsets[-1], :3] (-30 to 30)
    # view.add(scatter)
    #
    # view.camera = 'turntable'
    #
    # axis = visuals.XYZAxis(parent=view.scene)
    # vispy.app.run()


    # np.save("data/2011_09_26/clustered_markings/" + str(frame), markings_array)
    # print(frame, "done")
