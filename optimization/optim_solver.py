import numpy as np
from loguru import logger
import shutil
from util import IO
import yaml
import os
from matplotlib import pyplot as plt

from .sensor import *
from .geatpy_support import *


class OptimSolver:
    def __init__(self, name, system_settings, settings):
        self.sensor_list = system_settings["sensor_list"]
        self.output_yaml_path = system_settings["simu_input_dirname"]
        self.output_result_path = system_settings["visual_result_dirname"]
        self.workspace_path = system_settings["workspace_path"]
        self.name = name

        self.output_yaml_path = os.path.join(self.workspace_path, self.name, self.output_yaml_path)
        self.output_result_path = os.path.abspath(os.path.join(self.workspace_path, self.name, self.output_result_path))

        if not os.path.exists(self.output_result_path):
            os.makedirs(self.output_result_path)

        self.iter = 0
        self.state_logger = dict()

    def debug(self):
        pass

    def run(self, eval_result):
        logger.info("optimization module start to run")

    def setup(self):
        logger.info("optimization module start to set up")


class GeatpySupportedOptimSolver(OptimSolver):
    def __init__(self, name, system_settings, settings):
        super().__init__(name, system_settings, settings)
        self.dim = 0
        self.varTypes = []
        self.lb = []
        self.ub = []

        self.sensors = []
        for sensor_name in self.sensor_list:
            sensor = DE_OptimSolver.sensor_dict[sensor_name]()
            self.sensors.append(sensor)
            self.dim += sensor.dim
            self.varTypes += sensor.varTypes
            self.lb += sensor.lb
            self.ub += sensor.ub

    def update_logger(self, detail_report):
        self.__append_logger_elements("gen", self.iter)
        self.__add_logger_elements("phen", detail_report["phen"])
        for k in detail_report.keys():
            if k != "phen":
                self.__add_logger_elements(k, detail_report[k])
                self.__append_logger_elements(f"max_{k}", np.max(detail_report[k]))
                self.__append_logger_elements(f"min_{k}", np.min(detail_report[k]))
                self.__append_logger_elements(f"mean_{k}", np.mean(detail_report[k]))
        self.log()

    def log(self):
        without = ["gen", "phen"]
        for k in self.state_logger.keys():
            if ("max_" in k) or ("min_" in k) or ("mean_" in k):
                plt.figure()
                plt.plot(self.state_logger["gen"], self.state_logger[k])
                plt.savefig(os.path.join(self.output_result_path, f"gen--{k}.png"))
                plt.close()
            elif k not in without:
                plt.figure()
                plt.plot(self.state_logger["gen"], self.state_logger[f"max_{k}"])
                plt.plot(self.state_logger["gen"], self.state_logger[f"min_{k}"])
                plt.plot(self.state_logger["gen"], self.state_logger[f"mean_{k}"])
                plt.legend(["max", "min", "avg"])
                plt.savefig(os.path.join(self.output_result_path, f"gen--{k}.png"))
                plt.close()
        best_idx = np.argmax(self.state_logger['total'])
        self.write_best(self.state_logger['phen'][best_idx],self.output_result_path)
        with open(os.path.join(self.output_result_path,"state_logger.txt"),"w") as f:
            f.write(str(self.state_logger))
        logger.info(f"visual results of generation {self.iter} is output")

    def __add_logger_elements(self, key, elements):
        if key in self.state_logger.keys():
            self.state_logger[key] += elements
        else:
            self.state_logger[key] = elements

    def __append_logger_elements(self, key, elements):
        if key in self.state_logger.keys():
            self.state_logger[key].append(elements)
        else:
            self.state_logger[key] = [elements]

    def write(self, phen, path):
        # return names of yaml files
        # phen = self.algorithm.population.Phen
        if os.path.exists(self.output_yaml_path):
            shutil.rmtree(self.output_yaml_path)
        os.makedirs(self.output_yaml_path)
        return IO.write_configuration(phen, path, self.sensors)

    def write_best(self, phen, path):
        return IO.write_best_configuration(phen,self.iter, path, self.sensors)


class DE_OptimSolver(GeatpySupportedOptimSolver):
    sensor_dict = {
        "lidar": Lidar,
        "camera": Camera
    }

    def __init__(self, name, system_settings, settings):
        super().__init__(name, system_settings, settings)
        self.nand = settings["parameters"]["nand"]
        self.generation = settings["parameters"]["generation"]
        self.F = settings["parameters"]["F"]
        self.CR = settings["parameters"]["CR"]

        initial_population = ea.Population(
            Encoding='RI', NIND=self.nand
        )
        self.problem = SensorConfigurationProblem(
            self.dim, self.varTypes, self.lb, self.ub
        )
        self.algorithm = DE_currentToBest_1_L_online(
            self.problem, initial_population, self.F, self.CR, MAXGEN=self.generation
        )

        logger.info(
            f"optimization solver constructed [name: {self.name}, sensor list: {self.sensor_list}, nand: {self.nand}, generation: {self.generation}, F: {self.F}, CR: {self.CR}]")

    def setup(self):
        self.iter = 0
        self.algorithm.setup()
        self.problem.set_Fields(self.algorithm.population.Field)
        self.algorithm.population.Phen = self.algorithm.population.decoding()
        names_list = self.write(self.algorithm.population.Phen,self.output_yaml_path)
        logger.info("initial population (double sizes) is output and ready to simulate")
        return names_list, [s.result_suffix for s in self.sensors]

    def run(self, eval_result):
        self.iter += 1
        report_for_update = eval_result["report_for_update"]
        pops = report_for_update[0]
        fitness = report_for_update[1]
        self.problem.update_buffer(pops, fitness)
        self.algorithm.run_online()
        self.algorithm.population.Phen = self.algorithm.population.decoding()
        name_list = self.write(self.algorithm.population.Phen,self.output_yaml_path)
        logger.info(f"generation [{self.iter}] is generated")
        self.update_logger(eval_result["detail_report"])
        return name_list, [s.result_suffix for s in self.sensors]
