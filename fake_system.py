import os.path
import shutil
import time
import geatpy as ea
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from optimization.geatpy_support import *
from optimization.evolutionary_sensor import *
import pandas as pd
from main import BaseSolver


class FakeEvoSolver(BaseSolver):
    sensor_dict = {
        "lidar": Lidar,
        "camera": Camera
    }

    def __init__(self, runtime):
        super().__init__(runtime)
        self.buffered_data_path = "./collected_data/res.csv"
        self.recompute_total = True
        if self.recompute_total:
            self.weights = runtime["evaluation"]["weights"]
        self.recompute_evaluation = False

        self.name = "Fake_system_" + self.name + "_[" + ",".join(runtime["evaluation"]["method_list"]) + "]_" + \
                    str(runtime["optimization"]["parameters"]["nand"]) + "_" + \
                    str(runtime["optimization"]["parameters"]["generation"]) + "_" + \
                    str(runtime["optimization"]["parameters"]["F"]) + "_" + \
                    str(runtime["optimization"]["parameters"]["CR"]) + "_" + \
                    self.optimization_setting["name"] + f"_{time.time()}"
        self.iter = 0
        self.output_path = os.path.join(
            runtime["system"]["workspace_path"],
            self.name
        )
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)

        # set sensors
        self.sensor_list = self.system_setting["sensor_list"]
        self.sensor_setting = runtime["sensors"]
        self.sensors = []
        self.sensors_pos_idxes = []
        self.dim = 0
        self.varTypes = []
        self.lb = []
        self.ub = []
        cnt = 0
        for sensor_name in self.sensor_list:
            sensor = FakeEvoSolver.sensor_dict[sensor_name]()
            self.sensors.append(sensor)
            self.sensors_pos_idxes += [cnt, cnt + 1]
            self.dim += sensor.dim
            self.varTypes += sensor.varTypes
            self.lb += sensor.lb
            self.ub += sensor.ub
            cnt += sensor.dim
            for sensor_setting_tag in self.sensor_setting.keys():
                if sensor_name == sensor_setting_tag:
                    for k in self.sensor_setting[sensor_setting_tag]:
                        sensor.set_attribute_unsafe(k, self.sensor_setting[sensor_setting_tag][k])
                    break
        self.best_phen = []

        # set for EA
        self.nand = self.optimization_setting["parameters"]["nand"]
        self.generation = self.optimization_setting["parameters"]["generation"]
        self.F = self.optimization_setting["parameters"]["F"]
        self.CR = self.optimization_setting["parameters"]["CR"]

        initial_population = ea.Population(
            Encoding='RI', NIND=self.nand
        )
        self.problem = SensorConfigurationProblem(
            self.dim, self.varTypes, self.lb, self.ub
        )
        self.algorithm = DE_currentToBest_1_L_online(
            self.problem, initial_population, self.F, self.CR, MAXGEN=self.generation,logTras=False
        )
        self.algorithm.setup()
        self.problem.set_sensors(self.sensors)
        self.problem.set_Fields(self.algorithm.population.Field)
        self.algorithm.population.Phen = self.algorithm.population.decoding()

        self.state_logger = dict()
        # self.optimization_solver = DE_OptimSolver(self.name, self.system_setting, self.optimization_setting)
        # self.optimization_solver.set_sensors(self.sensors)

        logger.info("The system settings is loaded")

    def load_evaluation_buffer(self):
        df = pd.read_csv(self.buffered_data_path)
        self.columns = df.columns
        phen = df.iloc[:, :self.dim].to_numpy()
        if self.recompute_total:
            partial_result = df.iloc[:, self.dim:-1].to_numpy()
            weights = np.tile(self.weights, partial_result.shape[0]).reshape(partial_result.shape[0], -1)
            total = np.sum(partial_result * weights, axis=1)
            logger.info(f"max at\n {df.iloc[np.argmax(total), :]}")
            return phen, total
        else:
            total = df.iloc[:, -1].tonumpy().squeeze()
            return phen, total

    def run(self):
        self.problem.set_kth(1)
        phen, total = self.load_evaluation_buffer()
        self.problem.update_buffer(phen, total)
        #self.pos_fitness_relation(phen,total)
        for iteration in range(self.generation):
            logger.info(f"start generation {iteration}")
            experiment_pop = self.algorithm.run_online()
            experiment_pop.Phen = experiment_pop.decoding()
            self.log(iteration)
        best = pd.DataFrame(data=np.array(self.best_phen), columns=[str(i) for i in range(self.dim)] + ["total"])
        best.to_csv(os.path.join(self.output_path, "best.csv"), index=False)
        logger.info("All Finish")

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

    def log(self, iteration):
        self.__append_logger_elements("gen", iteration)
        objV = np.array(self.algorithm.population.ObjV)
        best_idx = np.argmax(objV)
        self.best_phen.append(list(self.algorithm.population.Phen[best_idx, :]) + [float(objV[best_idx])])
        self.__append_logger_elements("fitness_maximum", np.max(objV))
        self.__append_logger_elements("fitness_minimum", np.min(objV))
        self.__append_logger_elements("fitness_mean", np.mean(objV))

        logger.info(f"start to log for [{self.iter}]")
        plt.figure()
        plt.plot(self.state_logger["gen"], self.state_logger[f"fitness_maximum"])
        plt.plot(self.state_logger["gen"], self.state_logger[f"fitness_minimum"])
        plt.plot(self.state_logger["gen"], self.state_logger[f"fitness_mean"])
        plt.legend(["max", "min", "avg"])
        plt.savefig(os.path.join(self.output_path, f"evolution.png"))
        plt.close()

    def pos_fitness_relation(self, batched_phen, fitness):
        cnt = 0
        for sensor_idx, sensor in enumerate(self.sensors):
            logger.info(f"print pos-fitness relation figure for [{sensor_idx}] sensor")
            test_pos = np.array(batched_phen)[:, cnt:cnt + sensor.dim]
            cnt += sensor.dim

            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.set_xlim3d(-1.1, 0.25)
            ax.set_ylim3d(-0.4, 0.4)
            ax.set_zlim3d(1.45, 1.55)
            plt.gca().set_box_aspect(
                (np.max(sensor.valid_points[:, 0]) - np.min(sensor.valid_points[:, 0]),
                 np.max(sensor.valid_points[:, 1]) - np.min(sensor.valid_points[:, 1]),
                 np.max(sensor.valid_points[:, 2]) - np.min(sensor.valid_points[:, 2])))
            valid_places = sensor.valid_points

            shape = (400, 200)
            kth = 5
            x = np.linspace(-1.1, 0.25, shape[0])
            y = np.linspace(-0.4, 0.4, shape[1])
            X2, Y2 = np.meshgrid(x, -y)

            f = np.zeros(shape)
            z = np.zeros(shape)

            for i in range(shape[0]):
                for j in range(shape[1]):
                    dis = np.linalg.norm(test_pos[:, :2] - np.array([x[i], y[j]]), 2, axis=1)
                    idxes = np.argpartition(dis, kth=kth)[:kth].astype('int')
                    tmp_f = np.mean(np.array(fitness)[idxes])
                    f[i, j] = tmp_f

                    dis2 = np.linalg.norm(valid_places[:, :2] - np.array([x[i], y[j]]), 2, axis=1)
                    idxes2 = np.argpartition(dis2, kth=kth)[:kth].astype('int')
                    z[i, j] = np.mean(valid_places[idxes2, 2])

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
            plt.savefig(os.path.join(self.output_path, f"pos-fitness relation [{sensor_idx}]"))
            plt.close()

if __name__ == "__main__":
    import yaml

    with open("config/MCtest_single_camera.yaml", 'r') as f:
        runtime = yaml.load(f, yaml.FullLoader)
    fake_solver = FakeEvoSolver(runtime)
    fake_solver.run()
