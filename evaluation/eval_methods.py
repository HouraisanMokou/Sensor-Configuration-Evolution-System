"""
give each component of final evaluation
"""
import itertools
from collections import Counter
from util import util

import numpy as np
import cv2
from PIL import Image
from loguru import logger
import open3d as o3d


class EvaluationMethods:
    """
    basic class
    """

    def __init__(self):
        self.result_name = None

    def run(self, simu_ele):
        """
        :param simu_ele: {"phen":ndarry,"urls":[[[str]]]}
        :return: float, result of this method
        """

    def set_sensors(self, sensors):
        self.sensors = sensors

    def log_after_eval(self):
        logger.info(f'Method: [{self.result_name}] is finish for this population')


"""
===================
give a random value
===================
"""


class RandomEvaluation(EvaluationMethods):
    def __init__(self):
        super().__init__()
        self.result_name = "random"

    def run(self, simu_ele):
        return np.random.random()


"""
=======================================================
entropy of pixels at same position but different frames
=======================================================
"""


class TemporalEntropy(EvaluationMethods):
    """only for camera"""

    def __init__(self):
        super().__init__()
        self.result_name = "TE"

    def run(self, simu_ele):
        urls = simu_ele["urls"]
        data = None

        size = 256
        c = 0
        cnt = 0
        for scenario in urls:
            for sensor in scenario:
                if "png" in sensor[0]:
                    for url in sensor:
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
                    h = np.log(sigma_pix)
                    h = (h - np.min(h)) / (np.max(h) - np.min(h))
                    h = cv2.copyMakeBorder(h, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                    sums = []
                    for i in range(1, h.shape[0] - 1):
                        for j in range(1, h.shape[1] - 1):
                            around = [h[i - 1, j - 1], h[i - 1, j], h[i - 1, j + 1], h[i, j - 1], h[i + 1, j + 1],
                                      h[i + 1, j - 1], h[i + 1, j], h[i + 1, j + 1]]
                            if h[i, j] > np.mean(around) + 3 * np.std(around):
                                sums.append((i, j))
                    for i, j in sums:
                        h[i, j] = 0
                    c += np.mean(h)
        return c / cnt / 0.025


"""
=======================================================
entropy of pixels at different position and in same frames
=======================================================
"""


class PixEntropy(EvaluationMethods):
    """only for camera"""

    def __init__(self):
        super().__init__()
        self.result_name = "PixEN"

    def run(self, simu_ele):
        urls = simu_ele["urls"]

        camera_c = 0
        camera_cnt = 0
        for scenario in urls:
            for sensor in scenario:
                if "png" in sensor[0]:
                    cs = []
                    for url in sensor:
                        data = np.asarray(Image.open(url).convert('L'))
                        bins = np.bincount(data.flatten())
                        bins = bins[bins != 0]
                        p = bins / np.sum(bins)
                        h = -np.sum(p * np.log2(p))
                        cs.append(h)
                    c1 = np.mean(cs)
                    camera_c += c1
                    camera_cnt += 1
        return camera_c / camera_cnt / 0.06  # 0.06 is prior std


class ModifiedPixEntropy(EvaluationMethods):
    """only for camera"""

    def __init__(self):
        super().__init__()
        self.result_name = "PixEN"

    def run(self, simu_ele):
        urls = simu_ele["urls"]

        camera_c = 0
        camera_cnt = 0
        for scenario in urls:
            all_sensors_datas = []
            for sensor in scenario:
                if "png" in sensor[0]:
                    single_sensor_data = []
                    for url in sensor:
                        data = np.asarray(Image.open(url).convert('L'))
                        single_sensor_data.append(data.flatten())
                    all_sensors_datas.append(single_sensor_data)
            if len(all_sensors_datas) == 0:
                continue
            all_sensors_datas = np.array(all_sensors_datas)
            hs = []
            for slice_idx in range(all_sensors_datas.shape[1]):
                slice_data = all_sensors_datas[:, slice_idx, :].squeeze()
                bins = np.bincount(slice_data.flatten()).astype('float')
                bins = bins[bins != 0]
                p = bins / np.sum(bins)
                h = -np.sum(p * np.log2(p))
                hs.append(h)
            camera_c += np.mean(hs)
            camera_cnt += 1
        if camera_cnt == 0:
            return 0
        return camera_c / camera_cnt / 0.06  # 0.06 is prior std


class CameraCoverage(EvaluationMethods):
    def __init__(self):
        super().__init__()
        self.result_name = "coverage_camera"
        self.sensors = None
        self.point_file = "./config/points.txt"
        with open(self.point_file, "r") as f:
            self.points = np.array(eval(f.read()))
        x_lim = 50
        y_lim = 20
        z_lim = 4
        self.voxel_len = 0.1
        self.interest_space = np.array(
            list(itertools.product(np.arange(-x_lim / self.voxel_len, x_lim / self.voxel_len + 1),
                                   np.arange(-y_lim / self.voxel_len, y_lim / self.voxel_len + 1),
                                   np.arange(0, z_lim / self.voxel_len)))).astype(float)
        m1 = (self.interest_space[:, 0] > -1.5)
        m2 = (self.interest_space[:, 0] < 1.5)
        m3 = (self.interest_space[:, 1] > -0.5)
        m4 = (self.interest_space[:, 0] < 0.5)
        mask = m1
        for m in [m2, m3, m4]:
            mask = np.logical_and(mask, m)
        mask = np.logical_not(mask)
        self.interest_space = self.interest_space[mask, :]
        X = self.interest_space
        # scale = 120
        # sigma_x = 0.75 * scale
        # sigma_y = 1 * scale
        # self.weights = self.gaussian(X[:, 0] - np.mean(X[:, 0]), sigma_x) * self.gaussian(X[:, 1] - np.mean(X[:, 1]),
        #                                                                                   sigma_y)

        with open("./config/distribution.txt", 'r') as f:
            distribution_dict = eval(f.read())
            x_dis = np.array(distribution_dict['x'])
            y_dis = np.array(distribution_dict['y'])
            z_dis = np.array(distribution_dict['z'])
        x_w = np.log2(x_dis[X[:, 0].astype(int) + int(x_lim / self.voxel_len)] + 1)
        y_w = np.log2(y_dis[X[:, 1].astype(int) + int(y_lim / self.voxel_len)] + 1)
        z_w = np.log2(z_dis[X[:, 2].astype(int)] + 1)
        self.weights = x_w * y_w * z_w

        # self.interest_space = self.interest_space[
        #                       self.interest_space[:, 0] ** 2 + self.interest_space[:, 1] ** 2 + self.interest_space[:,
        #                                                                                         2] ** 2 > 16,
        #                       :]

    def gaussian(self, x, sigma):
        exp = np.exp(-1 / (2 * (sigma ** 2)) * (x ** 2))
        return exp

    def run(self, simu_ele):
        phen = simu_ele["phen"]
        cnt = 0

        total_mask = None
        offset = 0.05
        z_min = np.min(self.points[:, 2]) - offset
        x_max = np.max(self.points[:, 0]) + offset
        y_max = np.max(self.points[:, 1]) + offset
        y_min = np.min(self.points[:, 1]) - offset

        for sensor in self.sensors:
            if "png" in sensor.result_suffix:
                phen_slice = phen[cnt:cnt + sensor.dim]
                # if sensor.dim != 5:
                #     phen_slice = [phen_slice[0], phen_slice[1], 0, 0, 90]
                if sensor.dim == 2:
                    phen_slice = [phen_slice[0], phen_slice[1], 0, 0, 90]
                elif sensor.dim == 4:
                    phen_slice = [phen_slice[0], phen_slice[1], phen_slice[2], phen_slice[3], 90]
                sensor_z_pos = sensor.parameter_decompress(phen_slice[:2])[2]
                cnt += sensor.dim

                fov = phen_slice[-1]
                voxel_len = self.voxel_len

                usable = self.interest_space.copy()
                mask6 = (usable[:, 0] > phen_slice[0] / voxel_len)
                usable[np.logical_not(mask6), :] = 1
                div1 = usable[:, 0] - phen_slice[0] / voxel_len
                div1[div1 == 0] = 1e-8
                div2 = usable[:, 1] - phen_slice[1] / voxel_len
                div2[div2 == 0] = 1e-8
                tan1 = (usable[:, 1] - phen_slice[1] / voxel_len) / div1
                tan2 = (usable[:, 2] - sensor_z_pos / voxel_len) / div1
                tan3 = (usable[:, 2] - sensor_z_pos / voxel_len) / div2
                tans = np.vstack([tan1, tan2, tan3]).T
                half_fov = fov / 2
                lb_lateral = np.tan((phen_slice[2] - half_fov) * np.pi / 180)
                ub_lateral = np.tan((phen_slice[2] + half_fov) * np.pi / 180)
                lb_above = np.tan((-phen_slice[3] - half_fov) * np.pi / 180)
                lb_above2 = (z_min - sensor_z_pos) / (x_max - phen_slice[0])
                ub_above = np.tan((-phen_slice[3] + half_fov) * np.pi / 180)
                bound1 = (sensor_z_pos - z_min) / (y_max - phen_slice[1])
                bound2 = (sensor_z_pos - z_min) / (y_min - phen_slice[1])
                mask1 = (tans[:, 0] > lb_above)
                mask3 = (tans[:, 1] > lb_above2)
                mask2 = (tans[:, 1] > lb_lateral)
                mask4 = (tans[:, 0] < ub_above)
                mask5 = (tans[:, 1] < ub_lateral)
                mm = ((tans[:, 2] > 0) & (tans[:, 2] < bound1)) | ((tans[:, 2] < 0) & (tans[:, 2] > bound2))
                mask7 = mm | (usable[:, 2] > sensor_z_pos / voxel_len)
                mask = mask1
                for m in [mask2, mask3, mask4, mask5, mask6, mask7]:
                    mask = np.logical_and(mask, m)
                total_mask = mask.astype('float') if total_mask is None else total_mask + (mask.astype('float'))
        # total_mask = total_mask[total_mask != 0]
        # score = np.sum((2-0.5**(total_mask-1))/(2-0.5**(len(self.sensors)-1)))
        # score = np.sum(np.log2(1 + total_mask))
        if total_mask is None:
            return 0
        # q = 1 / 3  # q<1
        # score = np.sum(self.weights * (1 - q ** total_mask) / (1 - q))
        score = np.sum(self.weights * (np.log2(total_mask + 1)))
        return score / 0.06786838117594476  # 931260.7241582343 is prior std


class SSIM(EvaluationMethods):

    def __init__(self):
        super().__init__()
        self.result_name = "SSIM"

    def run(self, simu_ele):
        urls = simu_ele["urls"]
        meta = []
        c1 = 1e-4
        c2 = 1e-4
        score = []
        for scenario in urls:
            scenario_meta = []
            if len(scenario) == 1:
                return 0
            for sensor in scenario:
                if "png" in sensor[0]:
                    sensor_meta = []
                    for url in sensor:
                        data = np.asarray(Image.open(url).convert("L")).astype('float').flatten()
                        mu_data = np.mean(data)
                        std_data = np.std(data)
                        sensor_meta.append((data, mu_data, std_data))
                    scenario_meta.append(sensor_meta)
            scenario_res = []
            for idx1, sensor_meta1 in enumerate(scenario_meta):
                for idx2, sensor_meta2 in enumerate(scenario_meta):
                    if idx2 <= idx1:
                        continue
                    ssim_list = []
                    for slice_idx in range(len(sensor_meta1)):
                        meta1 = sensor_meta1[slice_idx]
                        meta2 = sensor_meta2[slice_idx]
                        cov = np.cov(meta1[0], meta2[0])[0, 1]
                        mu_x = meta1[1]
                        mu_y = meta2[1]
                        std_x = meta1[2]
                        std_y = meta2[2]
                        ssim = (2 * mu_x * mu_y + c1) * (2 * cov + c2) / (
                                (mu_x ** 2 + mu_y ** 2 + c1) * (std_x ** 2 + std_y ** 2 + c2))
                        ssim_list.append(ssim)
                    mean_ssim = np.mean(ssim_list)
                    scenario_res.append(mean_ssim)
            scenario_res = np.mean(scenario_res)
            score.append(scenario_res)
        score = - np.mean(score)  # ssim is higher, the configuration is more similar
        return score / 0.04  # 0.04 is prior std


class LidarCoverage(EvaluationMethods):
    def __init__(self):
        super().__init__()
        self.result_name = "coverage_lidar"
        self.sensors = None
        self.voxel_len = 0.1

    def run(self, simu_ele):
        urls = simu_ele["urls"]
        phen = simu_ele["phen"]
        phen_slices = []
        cnt = 0
        for sensor in self.sensors:
            phen_slice = phen[cnt:cnt + sensor.dim]
            cnt += sensor.dim
            phen_slices.append(phen_slice)

        scores = []
        for scenario in urls:
            slice_cnt = len(scenario[0])  # sensor cnt is not related with type of sensor
            for idx in range(slice_cnt):
                total_data = None
                for sensor, sensor_solver, phen_slice in zip(scenario, self.sensors, phen_slices):
                    if "ply" in sensor_solver.result_suffix:
                        data = o3d.io.read_point_cloud(sensor[idx])
                        data = np.array(data.points).T.astype(float)
                        x = phen_slice[0]
                        y = phen_slice[1]
                        z = sensor_solver.parameter_decompress(phen_slice[:2])[2]
                        t = np.array([[x], [y], [z]])
                        pitch = phen_slice[2]
                        # (x,y,pitch)
                        data = util.cloud_tf_inverse(data, t, 0, pitch, 0)
                        mask1 = np.logical_not(
                            (data[0, :] > -1.5) & (data[0, :] < 1.5) & (data[1, :] > -0.5) & (data[1, :] < 0.5)
                            )
                        data = data[:, mask1]
                        total_data = data if total_data is None else np.hstack([total_data, data])
                if total_data is None:
                    return 0
                voxels = np.round(total_data / self.voxel_len)
                voxels = list(map(tuple, voxels.T))
                counter = Counter(voxels)
                # count = np.array(list(counter.values()))
                score = len(list(counter.keys()))  # np.sum(np.log2(count + 1))
                scores.append(score)
        return np.mean(scores) / 500


class LidarPerceptionEntropy(EvaluationMethods):
    def __init__(self):
        super().__init__()
        self.result_name = "perception_entropy"
        self.sensors = None
        self.voxel_len = 0.1

    def run(self, simu_ele):
        a = 0.156
        b = 0.1
        urls = simu_ele["urls"]
        phen = simu_ele["phen"]
        phen_slices = []
        cnt = 0
        for sensor in self.sensors:
            phen_slice = phen[cnt:cnt + sensor.dim]
            cnt += sensor.dim
            phen_slices.append(phen_slice)

        scores = []
        for scenario in urls:
            slice_cnt = len(scenario[0])  # sensor cnt is not related with type of sensor
            for idx in range(slice_cnt):
                total_data = None
                for sensor, sensor_solver, phen_slice in zip(scenario, self.sensors, phen_slices):
                    if "ply" in sensor_solver.result_suffix:
                        data = o3d.io.read_point_cloud(sensor[idx])
                        data = np.array(data.points).T.astype(float)
                        x = phen_slice[0]
                        y = phen_slice[1]
                        z = sensor_solver.parameter_decompress(phen_slice[:2])[2]
                        t = np.array([[x], [y], [z]])
                        pitch = phen_slice[2]
                        # (x,y,pitch)
                        data = util.cloud_tf_inverse(data, t, 0, pitch, 0)
                        mask1 = np.logical_not(
                            (data[0, :] > -1.5) & (data[0, :] < 1.5) & (data[1, :] > -0.5) & (data[1, :] < 0.5)
                        )
                        data = data[:, mask1]
                        # mask2 = ((data[0, :] > -self.x_lim) and (data[0, :] < self.x_lim) and (
                        #             data[1, :] > -self.y_lim) and (data[1, :] < self.y_lim)) and (
                        #                     data[2, :] < self.z_lim)
                        total_data = data if total_data is None else np.hstack([total_data, data])
                if total_data is None:
                    return 0
                voxels = np.round(total_data / self.voxel_len)
                voxels = list(map(tuple, voxels.T))
                counter = Counter(voxels)
                count = np.array(list(counter.values()))
                ap = a * np.log(count) + b
                ap = np.array([_ for _ in ap if _ > 0])
                ap[ap > 1] = 1
                sigma = 1 / ap - 1
                sigma = [_ for _ in sigma if _ > 0]
                sigma = [_ for _ in sigma if not np.isnan(_)]
                score = np.mean(2 * np.log(sigma) + 1 + np.log(2 * np.pi))
                scores.append(-score)

        return np.mean(scores) / 0.08
