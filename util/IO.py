import numpy as np
import yaml
import os
import open3d as o3d
from PIL import Image

def write_best_configuration(p,iter, output_yaml_path, sensors):
    name_list = []
    cnt = 0
    yaml_dict = []
    for sensor in sensors:
        phen_slice = p[cnt:cnt + sensor.dim]
        cnt += sensor.dim
        yaml_dict.append(
            sensor.phen2yaml(phen_slice)
        )
    yaml_name = f"{iter}_best.yaml"
    name_list.append(yaml_name)
    with open(os.path.join(output_yaml_path, yaml_name), 'w') as f:
        yaml.dump(yaml_dict, f)

    return name_list

def write_configuration(phen, output_yaml_path, sensors):
    name_list = []
    for i, p in enumerate(list(phen)):
        cnt = 0
        yaml_dict = []
        for sensor in sensors:
            phen_slice = p[cnt:cnt + sensor.dim]
            cnt += sensor.dim
            yaml_dict.append(
                sensor.phen2yaml(phen_slice)
            )
        yaml_name = f"{str(i).rjust(4, '0')}_" + "_".join([str(_) for _ in p]) + ".yaml"
        name_list.append(yaml_name)
        with open(os.path.join(output_yaml_path, yaml_name), 'w') as f:
            yaml.dump(yaml_dict, f)
    return name_list


def read_data(url):
    if "ply" in url:
        data = o3d.io.read_point_cloud(url)
    elif "png" in url:
        data = Image.open(url)
        data = np.asarray(data)
    return data