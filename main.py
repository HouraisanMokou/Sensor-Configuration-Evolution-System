import argparse
import time
from loguru import logger
import yaml

from evaluation.eval_solver import *
from optimization.optim_solver import *
from simulation.SimuSolver import *
from optimization.evolutionary_sensor import *


class BaseSolver:
    def __init__(self, runtime):
        ## control the process of whole
        self.system_setting = runtime["system"]
        self.simulation_setting = runtime["simulation"]
        self.evaluation_setting = runtime["evaluation"]
        self.optimization_setting = runtime["optimization"]

        self.name = self.system_setting["name"]

    def run(self):
        pass


class EvolutionSolver(BaseSolver):  # geatpy support
    sensor_dict = {
        "lidar": Lidar,
        "camera": Camera,
        "defined-fov-camera": DefinedFovCamera
    }

    def __init__(self, runtime):
        super().__init__(runtime)
        self.name = self.name + "_[" + "_".join(runtime["evaluation"]["method_list"]) + "]_" + \
                    str(runtime["optimization"]["parameters"]["nand"]) + "_" + \
                    str(runtime["optimization"]["parameters"]["generation"]) + "_" + \
                    str(runtime["optimization"]["parameters"]["F"]) + "_" + \
                    str(runtime["optimization"]["parameters"]["CR"]) + "_" + \
                    self.optimization_setting["name"] + f"_{time.time()}"
        self.sensor_list = self.system_setting["sensor_list"]
        self.sensor_setting = runtime["sensors"]
        self.sensors = []
        for sensor_name in self.sensor_list:
            sensor = EvolutionSolver.sensor_dict[sensor_name]()
            self.sensors.append(sensor)
            for sensor_setting_tag in self.sensor_setting.keys():
                if sensor_name == sensor_setting_tag:
                    for k in self.sensor_setting[sensor_setting_tag]:
                        sensor.set_attribute_unsafe(k, self.sensor_setting[sensor_setting_tag][k])
                    break
        self.simu_solver = SimuSolver(self.name, self.system_setting, self.simulation_setting)
        self.evaluation_solver = ComplexEvalSolver(self.name, self.system_setting, self.evaluation_setting)
        self.optimization_solver = DE_OptimSolver(self.name, self.system_setting, self.optimization_setting)
        self.optimization_solver.set_sensors(self.sensors)
        self.evaluation_solver.set_sensors(self.sensors)
        logger.info("The system settings is loaded")

    def run(self):
        logger.info("setup initial population")
        population_meta = self.optimization_solver.setup()
        simu_report = self.simu_solver.setup(population_meta)
        eval_result = self.evaluation_solver.setup(simu_report)
        self.simu_solver.delete_data()

        for i in range(0, self.optimization_setting["parameters"]["generation"]):
            logger.info("Start of new generation")
            population_meta = self.optimization_solver.run(eval_result)
            simu_report = self.simu_solver.run(population_meta)
            eval_result = self.evaluation_solver.run(simu_report)
            self.simu_solver.delete_data()
            logger.info("End of this new generation")

        self.simu_solver.close()


if __name__ == '__main__':
    with open("./runtime.yaml", 'r') as f:
        runtime = yaml.load(f, yaml.FullLoader)
    # parser = argparse.ArgumentParser()
    # temporal setting
    # parser.add_argument("--name", default=runtime["system"]["name"], type=str)
    # parser.add_argument("--carla_path", default=runtime["simulation"]["carla_paths"], type=str)
    # parser.add_argument("--simu_result_dirname", default=runtime["system"]["simu_result_dirname"], type=str)
    # parser.add_argument("--slice_count", default=runtime["simulation"]["slice_count"], type=int)
    # parser.add_argument("--nand", default=runtime["optimization"]["parameters"]["nand"], type=int)
    # parser.add_argument("--generation", default=runtime["optimization"]["parameters"]["generation"], type=int)
    # args = parser.parse_args()
    # runtime["system"]["name"] = args.name
    # runtime["simulation"]["carla_paths"] = args.carla_path
    # runtime["system"]["simu_result_dirname"] = args.simu_result_path
    # runtime["simulation"]["slice_count"] = args.slice_count
    # runtime["optimization"]["parameters"]["nand"] = args.nand
    # runtime["optimization"]["parameters"]["generation"] = args.generation
    logger.info(f"Read configuration done.")

    solver = EvolutionSolver(runtime)
    solver.run()
