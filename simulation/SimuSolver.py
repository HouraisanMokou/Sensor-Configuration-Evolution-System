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

    def __init__(self, id,
                 CSCI_path,
                 carla_path,
                 port,
                 scenario_info,
                 input_path,
                 output_path,
                 rerun_input,
                 rerun_output, count,
                 wait_record_time):
        self.CSCI_path = CSCI_path
        self.id = id
        self.carla_path = carla_path
        self.port = port
        self.wait_record_time = wait_record_time
        self.input_path = input_path
        self.output_path = output_path
        self.rerun_input = rerun_input
        self.rerun_output = rerun_output
        self.count = count

        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)

        self.scenario_info = scenario_info
        self.start_carla_cmd = f"{self.carla_path} -carla-rpc-port={self.port}"
        self.start_CSCI_cmd = f"cd {self.CSCI_path} && python main.py --carla-port {self.port} -i {self.input_path} -o {self.output_path} -c {self.count}  -r{self.wait_record_time}"
        self.rerun_CSCI_cmd = f"cd {self.CSCI_path} && python main.py --carla-port {self.port} -i {self.rerun_input} -o {self.rerun_output} -c {self.count}  -r{self.wait_record_time}"
        self.carla_pid = None

    def set_iter(self, iter):
        self.start_CSCI_cmd = f"cd {self.CSCI_path} && python main.py --carla-port {self.port} -i {os.path.join(self.input_path, str(iter))} -o {os.path.join(self.output_path, str(iter))} -c {self.count}  -r{self.wait_record_time}"
        self.rerun_CSCI_cmd = f"cd {self.CSCI_path} && python main.py --carla-port {self.port} -i {os.path.join(self.rerun_input, str(iter))} -o {os.path.join(self.rerun_output, str(iter))} -c {self.count}  -r{self.wait_record_time}"

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

    def close_carla(self):
        if self.carla_pid is not None:
            subprocess.Popen("taskkill /F /T /PID " + str(self.carla_pid), shell=True)

    def setup(self):
        # renew scenario.yaml
        with open(os.path.join(self.CSCI_path, 'scenario.yaml'), 'w') as f:
            yaml.dump(self.scenario_info, f)
        logger.info("Simulation module has set up and started fort its first run.")
        self.system_call(self.start_CSCI_cmd, self.start_carla_cmd)

    def run(self):
        logger.info("Simulation module has started to simulate this generation")
        self.system_call(self.start_CSCI_cmd, self.start_carla_cmd)

    def rerun(self):
        logger.info("Simulation module has started to rerun this generation")
        self.system_call(self.rerun_CSCI_cmd, self.start_carla_cmd)

    def delete_data(self):
        logger.info("Simulation module is deleting simulate data")
        shutil.rmtree(os.path.join(self.output_path, str(iter)))

class SimuSolver:

    def __init__(self, name, system_settings, settings):
        self.sensor_list = system_settings["sensor_list"]
        self.input_yaml_path = system_settings["simu_input_dirname"]
        self.output_result_path = system_settings["simu_result_dirname"]
        self.rerun_input_path = system_settings["simu_broken_rerun_input_dirname"]
        self.rerun_output_path = system_settings["simu_broken_rerun_output_dirname"]
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
        self.rerun_input_path = os.path.abspath(os.path.join(self.workspace_path, self.name, self.rerun_input_path))
        self.rerun_output_path = os.path.abspath(os.path.join(self.workspace_path, self.name, self.rerun_output_path))

        self.iter = 0

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
                         self.rerun_input_path,
                         self.rerun_output_path,
                         self.count,
                         self.wait_record_time)
            )

    # check whether the number of simulation result is correct and return simulation report
    def check(self, path, phen, sensor_suffixes):
        '''
        :return: simulation_report={
            "broken_list": [] # population which fail to simulation
            "pop":[{
                "phen":[]
                "urls":[[[]]] # the number of scenario matrices with slice count columns, the number of sensors rows
            }...]
        }
        '''
        # names_list, sensor_suffixes = population_meta
        pops_dir = os.listdir(path)

        broken_list = []
        pops = []
        for pop_idx, pop_dir in enumerate(pops_dir):
            pop = np.array(phen[pop_idx]).astype('float')  # {"phen": np.array(pop_dir.split('_')[1:]).astype('float')}
            scenario_dir = os.listdir(os.path.join(path, pop_dir))
            scenario_dir.sort()
            if scenario_dir != self.scenario_name_list:
                logger.error(f"population [{pop_dir}] is broken")
                broken_list.append(pop_dir)
                continue
            scenario_url_matrix = []
            for scenario in scenario_dir:
                sensor_dir = os.listdir(os.path.join(path, pop_dir, scenario))
                if len(sensor_dir) != len(sensor_suffixes):
                    logger.error(f"population [{pop_dir}] is broken")
                    broken_list.append(pop_dir)
                    continue
                sensor_url_matrix = []
                for idx, sensor in enumerate(sensor_dir):
                    sensor_path = os.path.join(path, pop_dir, scenario, sensor)
                    result_name = os.listdir(sensor_path)
                    slice_url_vector = [
                        os.path.join(os.path.join(self.output_result_path, str(self.iter)), pop_dir, scenario, sensor,
                                     s) for s in
                        result_name]
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
        logger.info("simulation module start to run")
        self.iter += 1
        if not os.path.exists(os.path.join(self.output_result_path, str(self.iter))):
            os.makedirs(os.path.join(self.output_result_path, str(self.iter)))
        for task in self.simu_tasks:  # simulate
            task.set_iter(self.iter)
            task.run()
        res = self.check(os.path.join(self.output_result_path, str(self.iter)), population_meta[1],
                         population_meta[2])  # population_meta[1]: sensor_suffixes
        for task in self.simu_tasks:
            task.delete_data()
        # if len(res["broken_list"]) != 0:
        #     tmp_res = self.rerun(res["broken_list"], population_meta[1])
        #     res["pop"] += tmp_res["pop"]
        return res

    def setup(self, population_meta):
        logger.info("simulation module start to set up")
        if not os.path.exists(os.path.join(self.output_result_path, str(self.iter))):
            os.makedirs(os.path.join(self.output_result_path, str(self.iter)))
        for task in self.simu_tasks:  # simulate
            task.set_iter(0)
            task.setup()
        res = self.check(os.path.join(self.output_result_path, str(0)), population_meta[1], population_meta[2])
        for task in self.simu_tasks:
            task.delete_data()
        # if len(res["broken_list"]) != 0:
        #     tmp_res = self.rerun(res["broken_list"], population_meta[1])
        #     res["pop"] += tmp_res["pop"]
        return res

    def rerun(self, broken_list, sensor_suffixes, iter=0):
        logger.info(f"start to resimulate {len(broken_list)} files. [times: {iter + 1}]")
        if os.path.exists(self.rerun_input_path):
            shutil.rmtree(self.rerun_input_path)
        os.makedirs(self.rerun_input_path)
        for pop_dirname in broken_list:
            broken_yaml_path = os.path.join(self.input_yaml_path, f"{pop_dirname}.yaml")
            rerun_input_path = os.path.join(self.rerun_input_path, f"{pop_dirname}.yaml")
            shutil.copyfile(broken_yaml_path, rerun_input_path)
        for task in self.simu_tasks:
            task.rerun()
        res = self.check(self.rerun_output_path, sensor_suffixes)
        for broken_file in broken_list:
            try:
                shutil.rmtree(os.path.join(self.output_result_path, broken_file))
                shutil.copy(os.path.join(self.rerun_output_path, broken_file),
                            os.path.join(self.output_result_path, broken_file))
            except:
                logger.error("move rerun data fail")
        if len(res["broken_list"]) != 0:
            if iter < 10:
                tmp_res = self.rerun(res["broken"], sensor_suffixes, iter + 1)
                res["broken_list"] = tmp_res["broken_list"]
                res["pop"] += tmp_res["pop"]
            else:
                logger.error("simulation module has reran but failed too much times")
                assert 1 == 0, "simulation module has reran but failed too much times"
        else:
            return res

    def close(self):
        for task in self.simu_tasks:
            task.close_carla()
