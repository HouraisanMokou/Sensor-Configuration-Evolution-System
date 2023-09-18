import os.path
import shutil
import subprocess
import time

import numpy as np
import yaml
import pandas as pd

from optimization.evolutionary_sensor import *
from evaluation.eval_methods import *


class MonteCarloSample:
    sensor_dict = {
        "lidar": Lidar,
        "camera": Camera,
        "non_rotation_camera": NonRotationCamera
    }
    method_dict = {
        "camera_coverage": CameraCoverage,
        "temporal_entropy": TemporalEntropy,
        "pixel-level_entropy": PixEntropy,
        "random": RandomEvaluation,
        "ssim": SSIM
    }

    def __init__(self, runtime):
        sensor_list = runtime["system"]["sensor_list"]
        method_list = runtime["evaluation"]["method_list"]
        weights = runtime["evaluation"]["weights"]
        self.weights = np.array(weights)
        self.methods = []
        self.sensors = []
        self.dim = 0
        self.lb = []
        self.ub = []
        for sensor_name in sensor_list:
            sensor = MonteCarloSample.sensor_dict[sensor_name]()
            self.sensors.append(sensor)
            self.dim += sensor.dim
            self.lb += sensor.lb
            self.ub += sensor.ub
        for method in method_list:
            m = MonteCarloSample.method_dict[method]()
            m.set_sensors(self.sensors)
            self.methods.append(m)

        self.lb = np.array(self.lb)
        self.ub = np.array(self.ub)

        self.carla_pid = None
        self.port = runtime["simulation"]["ports"]
        self.carla_path = runtime["simulation"]["carla_paths"]
        self.CSCI_path = runtime["simulation"]["CSCI_path"]
        self.count = runtime["simulation"]["slice_count"]
        self.workspace=os.path.join(runtime["system"]["workspace_path"],"monte_carlo")
        self.input_path = os.path.abspath(f"{self.workspace}/{runtime['system']['simu_input_dirname']}")
        self.output_path = os.path.abspath(f"{self.workspace}/{runtime['system']['simu_result_dirname']}")
        self.start_carla_cmd = f"{self.carla_path} -carla-rpc-port={self.port}"
        self.start_CSCI_cmd = f"cd {self.CSCI_path} && python main.py --carla-port {self.port} -i {self.input_path} -o {self.output_path} -c {self.count}"
        self.report_path = "./collected_data/res.csv"

        self.simu = False  # True

    def run(self, batch=200):
        sampled = self.random_sample(batch)
        yaml_dicts = self.batched_phen2yaml(sampled)
        self.write_yaml_dicts(yaml_dicts, self.input_path)
        if self.simu:
            self.system_call(self.start_CSCI_cmd, self.start_carla_cmd)
        simu_report = self.collect_simu_info(sampled)
        eval_results = np.array(self.eval(simu_report))
        combined = np.hstack([sampled, eval_results])
        dataframe = pd.DataFrame(data=combined, columns=self.get_columns_name())
        dataframe.to_csv(self.report_path, index=False)

    def eval(self, simu_report):
        totals = []
        for simu_ele in simu_report["pop"]:
            total = 0
            scores = []
            for idx, method in enumerate(self.methods):
                score = method.run(simu_ele)
                total += score * self.weights[idx]
                scores.append(score)
            totals.append(scores + [total])
        return totals

    def random_sample(self, batch=200):
        sampled = np.random.uniform(0, 1, (batch, self.dim))
        sampled = sampled * np.tile(self.ub - self.lb, batch).reshape(batch, -1) \
                  + np.tile(self.lb, batch).reshape(batch, -1)
        return sampled

    def batched_phen2yaml(self, batched_phen):
        yaml_dicts = []
        for phen in list(batched_phen):
            cnt = 0
            yaml_dict = []
            for sensor in self.sensors:
                phen_slice = phen[cnt:cnt + sensor.dim]
                cnt += sensor.dim
                yaml_ele = sensor.phen2yaml(phen_slice)
                yaml_dict.append(yaml_ele)
            yaml_dicts.append(yaml_dict)
        return yaml_dicts

    def write_yaml_dicts(self, yaml_dicts, dirpath):
        if os.path.exists(dirpath):
            shutil.rmtree(dirpath)
        os.makedirs(dirpath)
        for idx, yaml_dict in enumerate(yaml_dicts):
            with open(os.path.join(dirpath, f"{idx}.yaml"), 'w') as f:
                yaml.dump(yaml_dict, f)
                f.close()

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

    def collect_simu_info(self, batched_phen):
        simu_report = {"broken_list": [], "pop": []}
        for idx, phen in enumerate(list(batched_phen)):
            path = os.path.join(self.output_path, str(idx))
            urls = []
            for scenario_dir in os.listdir(path):
                scenario_urls = []
                for sensor_dir in os.listdir(os.path.join(path, scenario_dir)):
                    sensor_urls = []
                    for url in os.listdir(os.path.join(path, scenario_dir, sensor_dir)):
                        sensor_urls.append(url)
                    scenario_urls.append(sensor_urls)
                urls.append(scenario_urls)
            simu_report["pop"].append({
                "phen": phen,
                "urls": urls
            })
        return simu_report

    def rerun_evaluation(self):
        '''
        only can be used when the structure of original file is not changed
        '''
        df = pd.read_csv(self.report_path)
        retest_phen = df.iloc[:, :self.dim].to_numpy()
        simu_report = self.collect_simu_info(retest_phen)
        eval_results = np.array(self.eval(simu_report))
        combined = np.hstack([retest_phen, eval_results])
        dataframe = pd.DataFrame(data=combined, columns=self.get_columns_name())
        dataframe.to_csv(self.report_path, index=False)

    def get_columns_name(self):
        attr_names = [str(i) for i in range(self.dim)]
        method_names = [m.result_name for m in self.methods]
        column_names = attr_names + method_names + ["total"]
        return column_names


with open("config/MCtest_double_camera.yaml", 'r') as f:
    yaml_dict=yaml.load(f,yaml.FullLoader)
    sampler = MonteCarloSample(yaml_dict)
    sampler.run()
# sampler.run(2)
# sampled = sampler.random_sample(2)
# sampler.write_yaml_dicts(sampler.batched_phen2yaml(sampled), "./workspace/monte_carlo")
