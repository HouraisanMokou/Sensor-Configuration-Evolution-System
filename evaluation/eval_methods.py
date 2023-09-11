"""
give each component of final evaluation
"""
import itertools
from collections import Counter

import numpy as np
from PIL import Image
from loguru import logger


class EvaluationMethods:
    """
    basic class
    """

    def run(self, simu_ele):
        """
        :param simu_ele: {"phen":ndarry,"urls":[[[str]]]}
        :return: float, result of this method
        """

    def set_sensors(self, sensors):
        self.sensors = sensors


"""
===================
give a random value
===================
"""


class RandomEvaluation(EvaluationMethods):
    def run(self, simu_ele):
        return np.random.random()


"""
=======================================================
entropy of pixels at same position but different frames
=======================================================
"""


class TemporalEntropy(EvaluationMethods):
    """only for camera"""

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
                    c += np.mean(h)
        return c / cnt


"""
=======================================================
entropy of pixels at different position and in same frames
=======================================================
"""


class PixEntropy(EvaluationMethods):
    """only for camera"""

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


class CameraCoverage(EvaluationMethods):
    def __init__(self):
        self.sensors = None
        self.point_file = "./config/points.txt"
        with open(self.point_file, "r") as f:
            self.points = np.array(eval(f.read()))
        x_lim = 50
        y_lim = 20
        z_lim = 4
        self.voxel_len = 0.1
        self.interest_space = np.array(
            list(itertools.product(np.arange(-x_lim / self.voxel_len, x_lim / self.voxel_len),
                                   np.arange(-y_lim / self.voxel_len, y_lim / self.voxel_len),
                                   np.arange(0, z_lim / self.voxel_len)))).astype(float)

        self.interest_space = self.interest_space[
                              self.interest_space[:, 0] ** 2 + self.interest_space[:, 1] ** 2 + self.interest_space[:,
                                                                                                2] ** 2 > 16,
                              :]

    def run(self, simu_ele):
        phen = simu_ele["phen"]
        cnt = 0

        total_mask = None
        for sensor in self.sensors:
            phen_slice = phen[cnt:cnt + sensor.dim]
            sensor_z_pos=sensor.parameter_decompress(phen_slice)[2]
            cnt += sensor.dim

            fov = phen_slice[-1]
            voxel_len = self.voxel_len
            z_min = np.min(self.points[:, 2])
            x_max = np.max(self.points[:, 0])

            usable = self.interest_space.copy()
            mask6 = (usable[:, 0] > phen[0] / voxel_len)
            usable[np.logical_not(mask6), :] = 1
            tan1 = (usable[:, 1] - phen_slice[1] / voxel_len) / (usable[:, 0] - phen_slice[0] / voxel_len)
            tan2 = (usable[:, 2] - sensor_z_pos / voxel_len) / (usable[:, 0] - phen_slice[0] / voxel_len)
            tans = np.vstack([tan1, tan2]).T
            half_fov = fov / 2
            lb_lateral = np.tan((-phen_slice[2] - half_fov) * np.pi / 180)
            ub_lateral = np.tan((phen_slice[2] + half_fov) * np.pi / 180)
            lb_above = np.tan((-phen_slice[3] - half_fov) * np.pi / 180)
            lb_above2 = (z_min - sensor_z_pos) / (x_max - phen_slice[0])
            ub_above = np.tan((phen_slice[3] + half_fov) * np.pi / 180)
            mask1 = (tans[:, 0] > lb_above)
            mask3 = (tans[:, 0] > lb_above2)
            mask2 = (tans[:, 1] > lb_lateral)
            mask4 = (tans[:, 0] < ub_above)
            mask5 = (tans[:, 1] < ub_lateral)
            mask = mask1
            for m in [mask2, mask3, mask4, mask5, mask6]:
                mask = np.logical_and(mask, m)
            total_mask = mask.astype('float') if total_mask is None else total_mask + (mask.astype('float'))
        # total_mask = total_mask[total_mask!=0]
        # score = np.sum((2-0.5**(total_mask-1))/(2-0.5**(len(self.sensors)-1)))
        score = np.sum(np.log2(1 + total_mask))
        return score / 917825.6544354344  # 931260.7241582343 is prior std
