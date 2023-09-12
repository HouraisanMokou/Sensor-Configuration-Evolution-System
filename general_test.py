import os
import shutil
import time

import pandas as pd
import yaml
import random
import numpy as np
import open3d as o3d
from optimization.sensor import Lidar, Camera
from collections import Counter
import itertools

import cv2

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import art3d
from PIL import Image

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib import offsetbox

import numba
from numba.core.types import List, Tuple
from numba.typed import Dict

workspace = './workspace'
test_file = './CSCI'
filename = f'general_'

shape = (10, 20)
with open('./config/points.txt', 'r') as f:
    points3 = np.array(eval(f.read()))

x = np.linspace(-1.1, 0.25, shape[0])
y = np.linspace(-0.4, 0.4, shape[1])

z = np.zeros(shape)
kth = 5
X, Y = np.meshgrid(x, -y)
print(X.shape)
lines = []
for i in range(shape[0]):
    for j in range(shape[1]):
        dis = np.linalg.norm(points3[:, :2] - np.array([x[i], y[j]]), 2, axis=1)
        idxes = np.argpartition(dis, kth=kth)[:kth]
        tmp_z = np.mean(points3[idxes, 2])
        z[i, j] = 1.7  # tmp_z + 0.3
        lines.append(tuple([x[i], y[j], z[i, j], 0, 0, 0]))
z = z.T

sensor_dict = {
    "lidar": Lidar(),
    "camera": Camera()
}
sensors_list = ["camera"]
sensors = [sensor_dict[_] for _ in sensors_list]

yamls = []
for line in lines:
    yaml_ele = []
    cnt = 0
    for i, sensor in enumerate(sensors):
        slice = line[cnt:cnt + 6]
        cnt += 6
        valid = 1
        sensor_dict = {
            'id': int(i),
            'valid': int(valid),
            'transform': {
                'x': float(slice[0]),
                'y': float(slice[1]),
                'z': float(slice[2]),
                'pitch': float(slice[3]),
                'yaw': float(slice[4]),  # float(rot[1]),
                'roll': float(slice[5]),  # float(rot[2]),
            },
            'blueprint_name': 'sensor.camera.rgb',
            'attribute': {
                "image_size_x": 1920,
                "image_size_y": 1920,
            }
        }
        yaml_ele.append(sensor_dict)
    yamls.append(yaml_ele)

if os.path.exists(f"./{workspace}/{filename}/input"):
    shutil.rmtree(f"./{workspace}/{filename}/input")
os.makedirs(f"./{workspace}/{filename}/input")
for i in range(len(yamls)):
    with open(f'./{workspace}/{filename}/input/{i}.yaml', 'w') as f:
        yaml.dump(yamls[i], f)

fitness = []


@numba.jit(numba.float64(numba.uint8[:, :]), nopython=True)
def edge_entropy(data):
    m = 3
    data = data / 255
    found = dict()
    xms = []
    Cs = []
    for i in range(0, data.shape[0] - m):
        for j in range(0, data.shape[1] - m):
            xm = data[j:j + m, i:i + m]
            if np.sum(xm) == 0:
                continue
            xms.append(xm)
    for xm1 in xms:
        # xm1_hash = hash(str(xm1.flatten))
        flatten_xm1 = xm1.flatten()
        xm1_hash = (flatten_xm1[0], flatten_xm1[1], flatten_xm1[2], flatten_xm1[3], flatten_xm1[4],
                    flatten_xm1[5], flatten_xm1[6], flatten_xm1[7], flatten_xm1[8])
        if xm1_hash in found:
            Cs.append(found[xm1_hash])
        cnt = 0
        for xm2 in xms:
            d = np.sum(np.logical_xor(xm1, xm2).astype(np.float64))
            a = np.sum(xm1)
            b = np.sum(xm2)
            m = a if a > b else b
            r = np.ceil(m * 0.5)
            if d < r:
                cnt += 1
        C = cnt / len(xms)
        found[xm1_hash] = C
        Cs.append(C)
    Cs = np.array(Cs)
    Cs = Cs[Cs != 0]
    phi1 = (1 / len(xms)) * np.sum(np.log2(Cs))

    m = 4
    found = dict()
    xms = []
    Cs = []
    for i in range(0, data.shape[0] - m):
        for j in range(0, data.shape[1] - m):
            xm = data[j:j + m, i:i + m]
            if np.sum(xm) == 0:
                continue
            xms.append(xm)
    for xm1 in xms:
        # xm1_hash = hash(str(xm1.flatten))
        flatten_xm1 = xm1.flatten()
        xm1_hash = (flatten_xm1[0], flatten_xm1[1], flatten_xm1[2], flatten_xm1[3], flatten_xm1[4],
                    flatten_xm1[5], flatten_xm1[6], flatten_xm1[7], flatten_xm1[8], flatten_xm1[9],
                    flatten_xm1[10], flatten_xm1[11], flatten_xm1[12], flatten_xm1[13], flatten_xm1[14],
                    flatten_xm1[15])
        if xm1_hash in found:
            Cs.append(found[xm1_hash])
        cnt = 0
        for xm2 in xms:
            d = np.sum(np.logical_xor(xm1, xm2).astype(np.float64))
            a = np.sum(xm1)
            b = np.sum(xm2)
            m = a if a > b else b
            r = np.ceil(m * 0.5)
            if d < r:
                cnt += 1
        C = cnt / len(xms)
        found[xm1_hash] = C
        Cs.append(C)
    Cs = np.array(Cs)
    Cs = Cs[Cs != 0]
    phi2 = (1 / len(xms)) * np.sum(np.log2(Cs))
    return phi1 - phi2


fov = 120
# interest_space_above = np.dstack(np.meshgrid(np.arange(-1000, 1000), np.arange(-200, 200))).reshape(-1, 2) * 0.1
# interest_space_lateral = np.dstack(np.meshgrid(np.arange(-1000, 1000), np.arange(0, 40))).reshape(-1, 2) * 0.1
t1 = time.time()
interest_space = np.array(list(itertools.product(np.arange(-500, 500), np.arange(-200, 200), np.arange(0, 40))))
print(time.time() - t1)


def cal_fitness(phen, path, id=0):
    # voxel_len = 0.1
    # usable = interest_space[interest_space[:, 0] > phen[0] / voxel_len, :]
    # tan1 = (usable[:, 1] - phen[1] / voxel_len) / (usable[:, 0] - phen[0] / voxel_len)
    # tan2 = (usable[:, 2] - phen[2] / voxel_len) / (usable[:, 0] - phen[0] / voxel_len)
    # tans = np.vstack([tan1, tan2]).T
    # half_fov = fov / 2
    # lb_lateral = np.tan(-phen[3] - half_fov)
    # ub_lateral = np.tan(phen[3] + half_fov)
    # lb_above = np.tan(-phen[4] - half_fov)
    # ub_above = np.tan(phen[4] + half_fov)
    # in_coverage = tans[tans[:, 0] > lb_above, :]
    # in_coverage = in_coverage[in_coverage[:, 0] < ub_above, :]
    # in_coverage = in_coverage[in_coverage[:, 1] > lb_lateral, :]
    # in_coverage = in_coverage[in_coverage[:, 1] < ub_lateral, :]
    # c = in_coverage.shape[0]
    # print(f"finish for {id}: [{phen}] [{c}]")
    urls = [os.path.join(path, f, s, i) for f in os.listdir(os.path.join(path)) for s in
            os.listdir(os.path.join(path, f)) for i in
            os.listdir(os.path.join(path, f, s))]
    print(urls)
    # if "png" in urls[0]:
    #     cs = []
    #     for idx, url in enumerate(urls):
    #         data = np.asarray(Image.open(url).convert('L'))
    #         bins = np.bincount(data.flatten())
    #         bins = bins[bins != 0]
    #         p = bins / np.sum(bins)
    #         h = -np.sum(p * np.log2(p))
    #         cs.append(h)
    #     c = np.mean(cs)
    size = 720
    c = 0
    cnt = 0
    data = None
    for url in urls:
        if data is None:
            data = np.asarray(Image.open(url).convert("L").resize((size, size)))[:, :, None]
        else:
            data = np.concatenate(
                [data, np.asarray(Image.open(url).convert("L").resize((size, size)))[:, :, None]],
                axis=2)
        cnt += 1
    data = data.astype("float")
    difference = np.abs(np.diff(data, axis=2))
    difference[difference == 0] = 1e-8
    ss = 1 / (difference ** 2)
    s = np.sum(ss, axis=2)
    sigma_pix = np.sqrt(1 / s)
    sigma_pix = cv2.resize(sigma_pix,(32,32))
    h = np.log(sigma_pix)
    # h = cv2.GaussianBlur(h, (3, 3), 50)
    h = (h - np.min(h)) / (np.max(h) - np.min(h))
    # h = cv2.resize(h, (32, 32))
    # h[h<0.3]=0
    h = cv2.copyMakeBorder(h,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
    sums =[]
    for i in range(1,h.shape[0]-1):
        for j in range(1,h.shape[1]-1):
            around=[h[i-1,j-1],h[i-1,j],h[i-1,j+1],h[i,j-1],h[i+1,j+1],h[i+1,j-1],h[i+1,j],h[i+1,j+1]]
            if h[i,j] > np.mean(around)+3*np.std(around):
                sums.append((i,j))
    for i,j in sums:
        h[i,j]=0
    c = np.mean(h)
    # print(np.min(h), np.max(h), np.mean(h), np.std(h))
    plt.imshow(h, cmap="gray")
    plt.show()
    c = c / cnt
    print(c)
    return c, (None, None)


def single_test():
    input = f"{workspace}/{filename}/input"
    output = f"{workspace}/{filename}/output"
    # os.system(
    #     f'cd {test_file} && python main.py -i {os.path.abspath(input)} -o {os.path.abspath(output)}')

    sensor_dir = input
    sensor_yamls = os.listdir(sensor_dir)
    clouds = []
    hs = []
    f1 = []
    f2 = []

    for i, sensor_yaml in enumerate(sensor_yamls):
        path = os.path.join(sensor_dir, sensor_yaml)
        with open(path, 'r') as f:
            sensor_dict = yaml.load(f, yaml.FullLoader)
            l = []
            for j in range(len(sensor_dict)):
                l += [sensor_dict[j]["transform"]["x"], sensor_dict[j]["transform"]["y"],
                      sensor_dict[j]["transform"]["z"],
                      sensor_dict[j]["transform"]["pitch"], sensor_dict[j]["transform"]["yaw"],
                      sensor_dict[j]["transform"]["roll"]]
            c, (c1, c2) = cal_fitness(l, os.path.join(output, str(i)), i)
            # hs.append(h)
            f1.append(c1)
            f2.append(c2)
            fitness.append(c)
    # print(f'f1:{np.max(f1)},{np.min(f1)},{np.mean(f1)},{np.std(f1)}')
    # print(f'f2:{np.max(f2)},{np.min(f2)},{np.mean(f2)},{np.std(f2)}')
    # idx=np.argmax(fitness)
    # idx2=np.argmin(fitness)
    # h=hs[idx]
    # print(idx,idx2,np.max(fitness),np.min(fitness))
    # plt.subplot(121)
    # plt.imshow(h)
    # plt.subplot(122)
    # plt.imshow(hs[idx2])
    # plt.show()

print(cal_fitness(None,"D:\projects\sensor_configuration\workspace\general_\output\\100"))
print(cal_fitness(None,"D:\projects\sensor_configuration\workspace\general_\output\\0"))
print(cal_fitness(None,"D:\projects\sensor_configuration\workspace\general_\output\\190"))
print(cal_fitness(None,"D:\projects\sensor_configuration\workspace\general_\output\\195"))
print(cal_fitness(None,"D:\projects\sensor_configuration\workspace\general_\output\\199"))

# single_test()
#
# fitness_dict = {}
# fitness = np.array(fitness)
#
# m = np.mean(fitness)
# n = np.std(fitness)
# print(m, n)
# # ub = np.mean(fitness) + 3 * np.std(fitness)
# # lb = np.mean(fitness) - 3 * np.std(fitness)
# # fitness[fitness > ub] = m
# # fitness[fitness < lb] = m
# # fitness = -fitness  # 要反转一下，强度图越大越好
# # fitness = 1/fitness#-np.sqrt(np.array(fitness))
# fitness = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness))
#
# for i, l in enumerate(lines):
#     fitness_dict[tuple(l)] = fitness[i]
#
# test = np.array([[x[i], y[j], z.T[i, j], fitness[i * shape[1] + j]] for i in range(shape[0]) for j in range(shape[1])])
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.set_xlim3d(-1.1, 0.25)
# ax.set_ylim3d(-0.4, 0.4)
# ax.set_zlim3d(1.45, 1.55)
# plt.gca().set_box_aspect((1.35, 0.8, 0.1))
#
# mm = -float('inf')
# mm_p = None
# f = np.zeros(shape)
# for i in range(shape[0]):
#     for j in range(shape[1]):
#         f[i, j] = fitness_dict[tuple([x[i], y[j], z.T[i, j], 0, 0, 0])]
#         if f[i, j] > mm:
#             mm = f[i, j]
#             mm_p = (i, j)
# print(mm, mm_p)
# f = f.T
#
# from matplotlib import cm
#
# f = (f - np.min(f)) / (np.max(f) - np.min(f))
# f = cm.jet(f)
# p = ax.plot_surface(X, Y, z, facecolors=f)
# m = cm.ScalarMappable(cmap=cm.jet)
# m.set_array(f)
# cax = fig.add_axes(
#     [ax.get_position().x1 + 0.01, ax.get_position().y0 + (ax.get_position().y1 - ax.get_position().y0) * 0.2, 0.02,
#      (ax.get_position().y1 - ax.get_position().y0) * 0.6])
#
# plt.colorbar(m, cax=cax)
# # ax.grid(False)
# # ax.axis('off')
# ax.view_init(30, -30)
# plt.show()

# shape = (400, 200)
# kth = 5
# x = np.linspace(-1.1, 0.25, shape[0])
# y = np.linspace(-0.4, 0.4, shape[1])
# X2, Y2 = np.meshgrid(x, -y)
# print(X2.shape)
# lines = []
# f = np.zeros(shape)
# z = np.zeros(shape)
#
# mm = -float('inf')
# mm_p = None
# for i in range(shape[0]):
#     for j in range(shape[1]):
#         dis = np.linalg.norm(test[:, :2] - np.array([x[i], y[j]]), 2, axis=1)
#         idxes = np.argpartition(dis, kth=kth)[:kth]
#         tmp_z = np.mean(test[idxes, 2])
#         z[i, j] = tmp_z
#
#         tmp_f = np.mean(test[idxes, 3])
#         f[i, j] = tmp_f
#
#         if f[i, j] > mm:
#             mm = f[i, j]
#             mm_p = (i, j)
# print(mm, mm_p)
#
# f = f.T
# z = z.T
#
# from matplotlib import cm
#
# f = (f - np.min(f)) / (np.max(f) - np.min(f))
# f = cm.jet(f)
# p = ax.plot_surface(X2, Y2, z, facecolors=f)
# m = cm.ScalarMappable(cmap=cm.jet)
# m.set_array(f)
# cax = fig.add_axes(
#     [ax.get_position().x1 + 0.01, ax.get_position().y0 + (ax.get_position().y1 - ax.get_position().y0) * 0.2, 0.02,
#      (ax.get_position().y1 - ax.get_position().y0) * 0.6])
#
# plt.colorbar(m, cax=cax)
# # ax.grid(False)
# # ax.axis('off')
# ax.view_init(30, -30)
# plt.savefig('res.png')
# plt.show()
