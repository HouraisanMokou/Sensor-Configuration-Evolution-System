import numpy as np


def cloud_tf_inverse(cloud, t, roll, pitch, yaw):
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(-pitch)
    yaw = np.deg2rad(-yaw)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, -np.sin(pitch)],
                   [0, 1, 0],
                   [np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    cloud = cloud + t
    cloud = np.linalg.pinv(Rx) @ cloud
    cloud = np.linalg.pinv(Ry) @ cloud
    cloud = np.linalg.pinv(Rz) @ cloud
    return cloud
