import numpy as np
import yaml
import shutil
from loguru import logger
import time
import subprocess

import os


class SimuTask:
    '''
    simulation task on single host
    '''

    def __init__(self, id, CSCI_path, carla_path, port, scenario_info, input_path, output_path, count,
                 wait_record_time):
        self.CSCI_path = CSCI_path
        self.id = id
        self.carla_path = carla_path
        self.port = port
        self.wait_record_time = wait_record_time
        self.input_path = input_path
        self.output_path = output_path
        self.count = count

        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)

        self.scenario_info = scenario_info
        self.start_carla_cmd = f"{self.carla_path} -carla-rpc-port={self.port}"
        self.start_CSCI_cmd = f"cd {self.CSCI_path} && python main.py --carla-port {self.port} -i {self.input_path} -o {self.output_path} -c {self.count}  -r{self.wait_record_time}"
        self.carla_pid = None

    def set_wait_record_time(self, wait_record_time):
        self.wait_record_time = wait_record_time
        self.start_CSCI_cmd = f"cd {self.CSCI_path} && python main.py --carla-port {self.port} -i {self.input_path} -o {self.output_path} -c {self.count}  -r{self.wait_record_time}"

    def system_call(self, CSCI_cmd, carla_cmd, cnt=0):
        res_signal = os.system(CSCI_cmd)
        if res_signal != 0:
            if cnt >= 10:
                logger.error("meets and fail to open much times")
                raise "fail to much times!"
            if self.carla_pid is not None:
                subprocess.Popen("taskkill /F /T /PID " + str(self.carla_pid), shell=True)
            time.sleep(2)
            obj = subprocess.Popen(carla_cmd)
            self.carla_pid = obj.pid
            time.sleep(2)

            self.system_call(CSCI_cmd, carla_cmd, cnt + 1)

    def setup(self):
        # renew scenario.yaml
        with open(os.path.join(self.CSCI_path, 'scenario.yaml')) as f:
            yaml.dump(self.scenario_info)
        logger.info("Simulation module has set up and started fort its first run.")
        self.system_call(self.start_CSCI_cmd, self.start_carla_cmd)

    def run(self):
        logger.info("Simulation module has start to simulate this generation")
        self.system_call(self.start_CSCI_cmd, self.start_carla_cmd)


class SimuSolver:

    def __init__(self, name, system_settings, settings):
        self.sensor_list = system_settings["sensor_list"]
        self.input_yaml_path = system_settings["simu_input_dirname"]
        self.output_result_path = system_settings["simu_result_dirname"]
        self.workspace_path = system_settings["workspace_path"]
        self.name = name

        self.carla_paths = settings["carla_paths"]
        self.posts = settings["ports"]
        self.slice_count = settings["slice_count"]
        self.scenario_info = settings["scenario_info"]
        self.wait_record_time = settings["wait_record_time"]
        self.CSCI_path = settings["CSCI_path"]
        self.count = settings["slice_count"]

        if len(self.posts) != len(self.carla_paths):
            logger.error("squeeze of posts and carla paths does not have same size")

        self.input_yaml_path = os.path.abspath(os.path.join(self.workspace_path, self.name, self.input_yaml_path))
        self.output_result_path = os.path.abspath(os.path.join(self.workspace_path, self.name, self.output_result_path))

        self.scenario_name_list = []
        for scenario in self.scenario_info:
            self.scenario_name_list.append(scenario["name"])
        self.scenario_name_list.sort()

        self.simu_tasks = []
        for i in range(len(self.posts)):
            self.simu_tasks.append(
                SimuTask(id, self.CSCI_path,
                         self.carla_paths[i],
                         self.posts[i],
                         self.scenario_info,
                         self.input_yaml_path,
                         self.output_result_path,
                         self.count,
                         self.wait_record_time)
            )

    # check whether the number of simulation result is correct and return simulation report
    def check(self, population_meta):
        '''
        :return: simulation_report={
            "broken_list": [] # population which fail to simulation
            "pop":[{
                "phen":[]
                "urls":[[[]]] # the number of scenario matrices with slice count columns, the number of sensors rows
            }...]
        }
        '''
        names_list, sensor_suffixes = population_meta
        pops_dir = os.listdir(self.output_result_path)

        broken_list = []
        pops = []
        for pop_dir in pops_dir:
            pop = {"phen": np.array(pop_dir.split('_')[1:]).astype('float')}
            urls = []
            scenario_dir = os.listdir(os.path.join(self.output_result_path, pop_dir))
            scenario_dir.sort()
            if scenario_dir != self.scenario_name_list:
                logger.error(f"population [{pop_dir}] is broken")
                broken_list.append(pop_dir)
                continue
            scenario_url_matrix = []
            for scenario in scenario_dir:
                sensor_dir = os.listdir(os.path.join(self.output_result_path, pop_dir, scenario))
                if len(sensor_dir) != len(sensor_suffixes):
                    logger.error(f"population [{pop_dir}] is broken")
                    broken_list.append(pop_dir)
                    continue
                sensor_url_matrix = []
                for idx, sensor in enumerate(sensor_dir):
                    sensor_path = os.path.join(self.output_result_path, pop_dir, scenario, sensor)
                    result_name = os.listdir(sensor_path)
                    slice_url_vector = [os.path.join(sensor_path, s) for s in result_name]
                    if len(result_name) != self.count:
                        logger.error(f"population [{pop_dir}] is broken")
                        broken_list.append(pop_dir)
                        continue
                    for name in result_name:
                        if sensor_suffixes[idx] not in name:
                            logger.error(f"population [{pop_dir}] is broken")
                            broken_list.append(pop_dir)
                            continue
                    sensor_url_matrix.append(slice_url_vector)
                scenario_url_matrix.append(sensor_url_matrix)
            pop["urls"] = scenario_url_matrix
            pops.append(pop)
        return {
            "broken_list": broken_list,
            "pop": pops
        }

    def run(self, population_meta):
        for task in self.simu_tasks:
            task.run()
        res=self.check(population_meta)
        logger.info("simulation module start to run")
        return res

    def setup(self, population_meta):
        for task in self.simu_tasks:
            task.setup()
        res=self.check(population_meta)
        logger.info("simulation module start to set up")
        return res
