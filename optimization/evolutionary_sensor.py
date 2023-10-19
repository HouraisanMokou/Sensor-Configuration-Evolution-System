import numpy as np
from loguru import logger


class Sensor:
    def __init__(self):
        self.points_path = "./config/points.txt"
        self.offset = 0
        self.min_rotation = -45
        self.max_rotation = 45

        with open(self.points_path, "r") as f:
            self.valid_points = np.array(eval(f.read()))

    def set_valid_points(self, points_path):
        self.points_path = points_path

        with open(self.points_path, "r") as f:
            self.valid_points = np.array(eval(f.read()))

    def set_attribute_unsafe(self, k, v):
        setattr(self, k, v)


'''
===========================
sensors for evolution system
===========================
'''


class EvolutionarySensor(Sensor):
    def __init__(self):
        super().__init__()
        self.blueprint_name = None

        self.kth = 5
        self.dim = None
        self.varTypes = None
        self.lb = None
        self.ub = None
        self.result_suffix = None

    def parameters_length_debug(self):
        if self.dim is None:
            logger.error(f"sensor {self.__class__}'s [dim] is None")
        if self.varTypes is None:
            logger.error(f"sensor {self.__class__}'s [varTypes] is None")
        if self.lb is None:
            logger.error(f"sensor {self.__class__}'s [lb] is None")
        if self.ub is None:
            logger.error(f"sensor {self.__class__}'s [ub] is None")
        if self.result_suffix is None:
            logger.error(f"sensor {self.__class__}'s [result_suffix] is None")
        if not (self.dim == len(self.varTypes) == len(self.lb) == len(self.ub)):
            logger.error(
                f"sensor {self.__class__}'s attributes has different length [{self.dim}, {len(self.varTypes)}, {len(self.lb)}, {len(self.ub)}]")

    def parameter_decompress(self, compressed):
        dis = np.linalg.norm(self.valid_points[:, :2] - compressed, 2, axis=1)
        idxes = np.argpartition(dis, kth=self.kth)[:self.kth]
        z_poses = np.mean(self.valid_points[idxes, 2] + self.offset)
        return np.hstack([compressed, z_poses])

    def parameter_compress(self, decompressed):
        return decompressed[:2]

    def decompose_attr(self, phen_slice):
        return None

    def compose_attr(self, attr_line):
        return []

    def decompose_pos(self, phen_slice):
        phen_slice = self.parameter_decompress(phen_slice[:2])
        return {"x": phen_slice[0], "y": phen_slice[1], "z": phen_slice[2]}

    def compose_pos(self, pos_line):
        return [pos_line["x"], pos_line["y"], pos_line["z"]]

    def phen2yaml(self, phen_slice):
        '''
        convert a phen slice to a sensor element in yaml
        '''
        phen_attr = self.decompose_attr(phen_slice)
        phen_pos = self.decompose_pos(phen_slice)
        yaml_ele = {
            "blueprint_name": self.blueprint_name,
            "transform": phen_pos
        }
        if phen_attr is not None:
            yaml_ele["attribute"] = phen_attr
        return yaml_ele

    def yaml2phen(self, yaml_ele):
        '''
        convert a sensor element in yaml to a phen slice
        '''
        phen_pos = self.decompose_pos(yaml_ele["transform"])
        phen_slice = self.parameter_compress(phen_pos)
        if "attribute" in yaml_ele.keys():
            phen_attr = self.compose_attr(yaml_ele["attribute"])
            phen_slice = phen_slice + phen_attr
        return phen_slice


class Lidar(EvolutionarySensor):
    '''
    phen:[x,y,pitch]
    '''

    def __init__(self):
        super().__init__()
        self.blueprint_name = "sensor.lidar.ray_cast"
        # attribute
        self.points_per_second = 60000
        self.min_rotation = -30
        self.max_rotation = 30

        self.dim = 3
        self.varTypes = [0, 0, 0]
        self.lb = [np.min(self.valid_points[:, 0]), np.min(self.valid_points[:, 1]), self.min_rotation]
        self.ub = [np.max(self.valid_points[:, 0]), np.max(self.valid_points[:, 1]), self.max_rotation]
        self.result_suffix = ".ply"

        self.parameters_length_debug()

    def decompose_attr(self, phen_slice):
        return {
            "points_per_second": self.points_per_second
        }

    def decompose_pos(self, phen_slice):
        pos = self.parameter_decompress(phen_slice[:2])
        pitch = phen_slice[2]
        return {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]), "pitch": float(pitch), "roll": 0, "yaw": 0}

    def compose_pos(self, pos_line):
        return [pos_line["x"], pos_line["y"], pos_line["pitch"]]


class Camera(EvolutionarySensor):
    '''
    phen:[x,y,pitch,yaw,fov]
    '''

    def __init__(self):
        super().__init__()
        self.blueprint_name = "sensor.camera.rgb"
        # rotation
        self.min_rotation_pitch = -15
        self.max_rotation_pitch = 15
        self.min_rotation_yaw = -30
        self.max_rotation_yaw = 30
        # attribute
        self.image_size_x = 720
        self.image_size_y = 720
        self.min_fov = 90
        self.max_fov = 120

        self.dim = 5
        self.varTypes = [0, 0, 0, 0, 0]
        self.lb = [np.min(self.valid_points[:, 0]),
                   np.min(self.valid_points[:, 1]),
                   self.min_rotation_pitch,
                   self.min_rotation_yaw,
                   self.min_fov]
        self.ub = [np.max(self.valid_points[:, 0]),
                   np.max(self.valid_points[:, 1]),
                   self.max_rotation_pitch,
                   self.max_rotation_yaw,
                   self.max_fov]
        self.result_suffix = ".png"

        self.parameters_length_debug()

    def decompose_attr(self, phen_slice):
        return {
            "image_size_x": self.image_size_x,
            "image_size_y": self.image_size_y,
            "fov": float(phen_slice[-1])
        }

    def compose_attr(self, attr_line):
        return [attr_line["fov"]]

    def decompose_pos(self, phen_slice):
        pos = self.parameter_decompress(phen_slice[:2])
        return {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]), "pitch": float(phen_slice[2]), "roll": 0,
                "yaw": float(phen_slice[3])}

    def compose_pos(self, pos_line):
        return [pos_line["x"], pos_line["y"], pos_line["pitch"], pos_line["yaw"]]


class NonRotationCamera(EvolutionarySensor):
    '''
    phen:[x,y]
    '''

    def __init__(self):
        super().__init__()
        self.blueprint_name = "sensor.camera.rgb"
        # rotation

        # attribute
        self.image_size_x = 1920
        self.image_size_y = 1920

        self.dim = 2
        self.varTypes = [0, 0]
        self.lb = [np.min(self.valid_points[:, 0]),
                   np.min(self.valid_points[:, 1])]
        self.ub = [np.max(self.valid_points[:, 0]),
                   np.max(self.valid_points[:, 1])]
        self.result_suffix = ".png"

        self.parameters_length_debug()

    def decompose_attr(self, phen_slice):
        return {
            "image_size_x": self.image_size_x,
            "image_size_y": self.image_size_y,
        }

    def decompose_pos(self, phen_slice):
        pos = self.parameter_decompress(phen_slice[:2])
        return {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]), "pitch": 0, "roll": 0, "yaw": 0}

    def compose_pos(self, pos_line):
        return [pos_line["x"], pos_line["y"]]


class DefinedFovCamera(EvolutionarySensor):
    '''
    phen:[x,y,pitch,yaw,fov]
    '''

    def __init__(self):
        super().__init__()
        self.blueprint_name = "sensor.camera.rgb"
        # rotation
        self.min_rotation_pitch = -15
        self.max_rotation_pitch = 15
        self.min_rotation_yaw = -30
        self.max_rotation_yaw = 30
        # attribute
        self.image_size_x = 720
        self.image_size_y = 720

        self.dim = 4
        self.varTypes = [0, 0, 0, 0]
        self.lb = [np.min(self.valid_points[:, 0]),
                   np.min(self.valid_points[:, 1]),
                   self.min_rotation_pitch,
                   self.min_rotation_yaw]
        self.ub = [np.max(self.valid_points[:, 0]),
                   np.max(self.valid_points[:, 1]),
                   self.max_rotation_pitch,
                   self.max_rotation_yaw]
        self.result_suffix = ".png"

        self.parameters_length_debug()

    def decompose_attr(self, phen_slice):
        return {
            "image_size_x": self.image_size_x,
            "image_size_y": self.image_size_y,
        }

    def decompose_pos(self, phen_slice):
        pos = self.parameter_decompress(phen_slice[:2])
        return {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]), "pitch": float(phen_slice[2]), "roll": 0,
                "yaw": float(phen_slice[3])}

    def compose_pos(self, pos_line):
        return [pos_line["x"], pos_line["y"], pos_line["pitch"], pos_line["yaw"]]
