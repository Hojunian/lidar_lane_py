import numpy as np
import vispy.scene
import sys
import os
import random
import math
import ransort as rs
import cv2
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

lane_root_dir = "data/2011_09_26/lane_points/"
lane_filename = os.listdir(lane_root_dir)
lane_filename.sort()

dot_root_dir = "data/2011_09_26/dot_lane_points/"
dot_filename = os.listdir(dot_root_dir)
dot_filename.sort()

cubic = PolynomialFeatures(degree=3)

def add_square_feature(X):
    X = np.concatenate([(X**2).reshape(-1,1), X], axis=1)
    return X

for file in lane_filename:
    points = np.asarray(np.load(lane_root_dir + file)).reshape((-1, 3))
    x = points[:, 0].reshape(-1, 1)
    y = points[:, 1].reshape(-1, 1)
    z = points[:, 2].reshape(-1, 1)

    ransac = linear_model.RANSACRegressor(residual_threshold=0.15)
    ransac.fit(cubic.fit_transform(x), y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models
    line_X = np.arange(x.min(), x.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(cubic.fit_transform(line_X))

    # Compare estimated coefficients
    print("Estimated coefficients (RANSAC):")
    print(ransac.estimator_.coef_)


    lw = 2
    plt.scatter(x[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
                label='Inliers')
    plt.scatter(x[outlier_mask], y[outlier_mask], color='gold', marker='.',
                label='Outliers')
    plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
             label='RANSAC regressor')
    plt.legend(loc='lower right')
    plt.xlabel("Input")
    plt.ylabel("Response")
    plt.ylim(-12.5,15)
    plt.show()
    break
    ransac2 = linear_model.RANSACRegressor(residual_threshold=0.15)
    ransac2.fit(cubic.fit_transform(x), z)
    inlier_mask = ransac2.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models
    line_X = np.arange(x.min(), x.max())[:, np.newaxis]
    line_z_ransac = ransac2.predict(cubic.fit_transform(line_X))

    # Compare estimated coefficients
    print("Estimated coefficients (RANSAC):")
    print(ransac2.estimator_.coef_)

    lw = 2
    # plt.scatter(x[inlier_mask], z[inlier_mask], color='yellowgreen', marker='.',
    #             label='Inliers')
    # plt.scatter(x[outlier_mask], z[outlier_mask], color='gold', marker='.',
    #             label='Outliers')
    # plt.plot(line_X, line_z_ransac, color='cornflowerblue', linewidth=lw,
    #          label='RANSAC regressor')
    # plt.legend(loc='lower right')
    # plt.xlabel("Input")
    # plt.ylabel("Response")
    # plt.show()
    lane_coeffs = []
    lane_coeffs.append(ransac.estimator_.coef_)
    lane_coeffs.append(ransac2.estimator_.coef_)
    lane_coeffs.append(np.asarray([[x.min(), x.max(),x.min(), x.max()]]))
    lane_coeffs = np.asarray(lane_coeffs).reshape(3, 4)
    np.save("data/2011_09_26/lane_eq/" + file, lane_coeffs)

for file in dot_filename:
    points = np.asarray(np.load(dot_root_dir + file)).reshape((-1, 3))
    x = points[:, 0].reshape(-1, 1)
    y = points[:, 1].reshape(-1, 1)
    z = points[:, 2].reshape(-1, 1)

    ransac = linear_model.RANSACRegressor(residual_threshold=0.15)
    ransac.fit(cubic.fit_transform(x), y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models
    line_X = np.arange(x.min(), x.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(cubic.fit_transform(line_X))

    # Compare estimated coefficients
    print("Estimated coefficients (RANSAC):")
    print(ransac.estimator_.coef_)


    lw = 2
    plt.scatter(x[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
                label='Inliers')
    plt.scatter(x[outlier_mask], y[outlier_mask], color='gold', marker='.',
                label='Outliers')
    plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
             label='RANSAC regressor')
    plt.legend(loc='lower right')
    plt.xlabel("Input")
    plt.ylabel("Response")
    plt.ylim(-12.5,15)
    plt.show()

    ransac2 = linear_model.RANSACRegressor(residual_threshold=0.15)
    ransac2.fit(cubic.fit_transform(x), z)

    inlier_mask = ransac2.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models
    line_X = np.arange(x.min(), x.max())[:, np.newaxis]
    line_z_ransac = ransac2.predict(cubic.fit_transform(line_X))

    # Compare estimated coefficients
    print("Estimated coefficients (RANSAC):")
    print(ransac2.estimator_.coef_)

    lw = 2
    # plt.scatter(x[inlier_mask], z[inlier_mask], color='yellowgreen', marker='.',
    #             label='Inliers')
    # plt.scatter(x[outlier_mask], z[outlier_mask], color='gold', marker='.',
    #             label='Outliers')
    # plt.plot(line_X, line_z_ransac, color='cornflowerblue', linewidth=lw,
    #          label='RANSAC regressor')
    # plt.legend(loc='lower right')
    # plt.xlabel("Input")
    # plt.ylabel("Response")
    # plt.show()
    lane_coeffs = []
    lane_coeffs.append(ransac.estimator_.coef_)
    lane_coeffs.append(ransac2.estimator_.coef_)
    lane_coeffs.append(np.asarray([[x.min(), x.max(),x.min(), x.max()]]))
    lane_coeffs = np.asarray(lane_coeffs).reshape(3, 4)
    np.save("data/2011_09_26/dot_lane_eq/" + file, lane_coeffs)
