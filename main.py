import argparse
import time
from loguru import logger
import yaml

from evaluation.eval_solver import *
from optimization.optim_solver import *
from simulation.SimuSolver import *


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


class EvolutionSolver(BaseSolver):

    def __init__(self, runtime):
        super().__init__(runtime)
        self.name = self.system_setting["name"] + "_" + self.optimization_setting["name"]
        self.simu_solver = SimuSolver(self.name, self.system_setting, self.simulation_setting)
        self.evaluation_solver = TE_EvalSolver(self.name, self.system_setting, self.evaluation_setting)
        self.optimization_solver = DE_OptimSolver(self.name, self.system_setting, self.optimization_setting)

    def run(self):
        logger.info("setup initial population")
        population_meta = self.optimization_solver.setup()
        simu_report = self.simu_solver.setup(population_meta)
        eval_result = self.evaluation_solver.setup(simu_report)

        for i in range(1, self.optimization_setting["parameters"]["generation"]):
            population_meta = self.optimization_solver.run(eval_result)
            simu_report = self.simu_solver.run(population_meta)
            eval_result = self.evaluation_solver.run(simu_report)


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
    runtime["system"]["name"] += f"_{time.time()}"
    logger.info(f"Read configuration done.")

    solver = EvolutionSolver(runtime)
    solver.run()
