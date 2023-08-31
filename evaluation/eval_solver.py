import random

import numpy as np
from loguru import logger
import os
from matplotlib import pyplot as plt
from PIL import Image
import open3d as o3d


class EvalSolver:
    def __init__(self, name, system_settings, settings):
        self.name = name
        self.sensor_list = system_settings["sensor_list"]
        self.input_result_path = system_settings["simu_result_dirname"]
        self.output_result_path = system_settings["visual_result_dirname"]
        self.workspace_path = system_settings["workspace_path"]

        self.input_result_path = os.path.abspath(os.path.join(self.workspace_path, self.name, self.input_result_path))

    def eval_pop(self, simu_ele):
        return dict()

    def setup(self, simu_report):
        return self.run(simu_report)

    def run(self, simu_report):
        detail_report = dict()
        report_for_update = [[], []]  # [[phen],[fitness]]
        for simu_ele in simu_report["pop"]:
            res = self.eval_pop(simu_ele)
            for k in res.keys():
                if k in detail_report.keys():
                    detail_report[k].append(res[k])
                else:
                    detail_report[k] = [res[k]]
            report_for_update[0].append(simu_ele["phen"])
            report_for_update[1].append(res["total"])
        return {
            "detail_report": detail_report,
            "report_for_update": report_for_update
        }


'''
================================================
give evaluation result a random value
================================================
'''


class Random_EvalSolver(EvalSolver):

    def __init__(self, name, system_settings, settings):
        super().__init__(name, system_settings, settings)

    def eval_pop(self, simu_ele):
        r1 = np.random.random()
        r2 = np.random.random()
        return {
            "phen": simu_ele["phen"],
            "total": r1 + r2,
            "perception_entropy": r1,
            "field_score": r2
        }


'''
================================================
evaluate with perception entropy and field score
================================================
'''

'''
================================================
evaluation with temporal entropy
================================================
'''


class TE_EvalSolver(EvalSolver):

    def __init__(self, name, system_settings, settings):
        super().__init__(name, system_settings, settings)

    def eval_pop(self, simu_ele):
        urls = simu_ele["urls"]
        data = None

        size = 256
        camera_c = 0
        lidar_c = 0
        camera_cnt = 0
        lidar_cnt = 0
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
                    camera_cnt += 1
                    data = data.astype("float")
                    difference = np.abs(np.diff(data, axis=2))
                    difference[difference == 0] = 1e-8
                    ss = 1 / (difference ** 2)
                    s = np.sum(ss, axis=2)
                    sigma_pix = np.sqrt(1 / s)
                    h = np.log(sigma_pix)
                    camera_c += np.mean(h)
                elif "ply" in sensor[0]:
                    continue
                    # for url in sensor:
                    #     if data is None:
                    #         data = np.asarray(Image.open(url).convert("L").resize((size, size)))[:, :, None]
                    #     else:
                    #         data = np.concatenate(
                    #             [data, np.asarray(Image.open(url).convert("L").resize((size, size)))[:, :, None]],
                    #             axis=2)
                    # camera_cnt += 1
                    # data = data.astype("float")
                    # difference = np.abs(np.diff(data, axis=2))
                    # difference[difference == 0] = 1e-8
                    # ss = 1 / (difference ** 2)
                    # s = np.sum(ss, axis=2)
                    # sigma_pix = np.sqrt(1 / s)
                    # h = np.log(sigma_pix)
                    # camera_c += np.mean(h)

        if camera_cnt != 0 and lidar_cnt != 0:
            c = camera_c / camera_cnt + lidar_c / lidar_cnt
        elif camera_cnt == 0:
            c = lidar_c / lidar_cnt
        else:
            c = camera_c / camera_cnt
        return {
            "phen": simu_ele["phen"],
            "total": c
        }
