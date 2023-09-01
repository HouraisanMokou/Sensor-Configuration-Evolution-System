import numpy as np
from loguru import logger
import shutil
from util import IO
import yaml
import os
from matplotlib import pyplot as plt
from matplotlib import cm

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
        self.sensors_pos_idxes = []
        cnt = 0
        for sensor_name in self.sensor_list:
            sensor = DE_OptimSolver.sensor_dict[sensor_name]()
            self.sensors_pos_idxes += [cnt, cnt + 1]
            self.sensors.append(sensor)
            self.dim += sensor.dim
            self.varTypes += sensor.varTypes
            self.lb += sensor.lb
            self.ub += sensor.ub
            cnt += sensor.dim

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
        self.write_best(self.state_logger['phen'][best_idx], self.output_result_path)
        with open(os.path.join(self.output_result_path, "state_logger.txt"), "w") as f:
            f.write(str(self.state_logger))
        if self.iter % 5 == 0:
            logger.info(f"print pos-fitness relation figure for first sensor at [generation {self.iter}]")

            test_pos = np.array(self.state_logger["phen"])[:,self.sensors_pos_idxes]
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.set_xlim3d(-1.1, 0.25)
            ax.set_ylim3d(-0.4, 0.4)
            ax.set_zlim3d(1.45, 1.55)
            plt.gca().set_box_aspect(
                (np.max(self.sensors[0].valid_points[:, 0]) - np.min(self.sensors[0].valid_points[:, 0]),
                 np.max(self.sensors[0].valid_points[:, 1]) - np.min(self.sensors[0].valid_points[:, 1]),
                 np.max(self.sensors[0].valid_points[:, 2]) - np.min(self.sensors[0].valid_points[:, 2])))
            shape = (400, 200)
            kth = 5
            x = np.linspace(-1.1, 0.25, shape[0])
            y = np.linspace(-0.4, 0.4, shape[1])
            X2, Y2 = np.meshgrid(x, -y)
            f = np.zeros(shape)
            z = np.zeros(shape)

            for i in range(shape[0]):
                for j in range(shape[1]):
                    dis = np.linalg.norm(test_pos[:,:2] - np.array([x[i], y[j]]), 2, axis=1)
                    idxes = np.argpartition(dis, kth=kth)[:kth].astype('int')
                    tmp_z=0
                    for idx in idxes:
                        tmp_z+=self.sensors[0].parameter_decompress(test_pos[idx,:])[2]
                    tmp_z=tmp_z/len(idxes)
                    z[i, j] = tmp_z

                    tmp_f = np.mean(np.array(self.state_logger["total"])[idxes])
                    f[i, j] = tmp_f

            f = f.T
            z = z.T

            f = (f - np.min(f)) / (np.max(f) - np.min(f))
            f = cm.jet(f)
            p = ax.plot_surface(X2, Y2, z, facecolors=f)
            m = cm.ScalarMappable(cmap=cm.jet)
            m.set_array(f)
            cax = fig.add_axes(
                [ax.get_position().x1 + 0.01,
                 ax.get_position().y0 + (ax.get_position().y1 - ax.get_position().y0) * 0.2, 0.02,
                 (ax.get_position().y1 - ax.get_position().y0) * 0.6])

            plt.colorbar(m, cax=cax)
            ax.grid(False)
            ax.axis('off')
            ax.view_init(30, -30)
            plt.savefig(os.path.join(self.output_result_path, "pos-fitness relation"))
            plt.close()
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
        return IO.write_best_configuration(phen, self.iter, path, self.sensors)


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
        names_list = self.write(self.algorithm.population.Phen, self.output_yaml_path)
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
        name_list = self.write(self.algorithm.population.Phen, self.output_yaml_path)
        logger.info(f"generation [{self.iter}] is generated")
        self.update_logger(eval_result["detail_report"])
        return name_list, [s.result_suffix for s in self.sensors]
