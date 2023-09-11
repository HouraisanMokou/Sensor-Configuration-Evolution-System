import random

import numpy as np
from loguru import logger
import os
from matplotlib import pyplot as plt
from PIL import Image
import open3d as o3d
from .eval_methods import *


class EvalSolver:
    def __init__(self, name, system_settings, settings):
        self.name = name
        self.sensor_list = system_settings["sensor_list"]
        self.input_result_path = system_settings["simu_result_dirname"]
        self.output_result_path = system_settings["visual_result_dirname"]
        self.workspace_path = system_settings["workspace_path"]

        self.input_result_path = os.path.abspath(os.path.join(self.workspace_path, self.name, self.input_result_path))
        self.sensors = None
        self.eval_methods = None

    def set_sensors(self, sensors):
        self.sensors = sensors
        if self.eval_methods is not None:
            for eval_method in self.eval_methods:
                eval_method.set_sensors(sensors)

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
        self.eval_methods = [
            TemporalEntropy()
        ]

    def eval_pop(self, simu_ele):
        score = self.eval_methods[0].run(simu_ele)
        return {
            "phen": simu_ele["phen"],
            "total": score
        }


class PixEN_EvalSolver(EvalSolver):

    def __init__(self, name, system_settings, settings):
        super().__init__(name, system_settings, settings)
        self.eval_methods = [
            PixEntropy(),
            CameraCoverage()
        ]

    def eval_pop(self, simu_ele):
        pixen_score = self.eval_methods[0].run(simu_ele)
        coverage_score = self.eval_methods[1].run(simu_ele)
        return {
            "phen": simu_ele["phen"],
            "total": pixen_score + coverage_score,
            "pixEn": pixen_score,
            "coverage": coverage_score
        }


class CameraCoverage_EvalSolver(EvalSolver):

    def __init__(self, name, system_settings, settings):
        super().__init__(name, system_settings, settings)
        self.eval_methods = [
            CameraCoverage()
        ]

    def eval_pop(self, simu_ele):
        coverage_score = self.eval_methods[0].run(simu_ele)
        return {
            "phen": simu_ele["phen"],
            "total": coverage_score,
        }
