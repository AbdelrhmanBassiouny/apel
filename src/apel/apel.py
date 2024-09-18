import json
import os.path
import shutil

import numpy as np
from tf.transformations import quaternion_from_matrix
from typing_extensions import List, Dict

from pycram.datastructures.pose import Pose
from pycram.datastructures.world import World
from pycram.datastructures.enums import ObjectType
from pycram.world_concepts.world_object import Object


class APEL:
    """
    A class used to load annotated perceived environments to PyCRAM.
    """
    resources_path: str = "../resources"
    envs_path: str = os.path.join(resources_path, "envs")

    def __init__(self, json_file: str, world: World):
        """
        Initializes the APEL class.

        :param json_file: The json file that contains the annotated perceived environment.
        :param world: The world to load the objects into.
        """
        self.json_file: str = json_file
        self.env_dir: str = os.path.dirname(json_file)
        self.data: List[Dict] = self.load_json_file()
        self.pycram_objects: List[Object] = []
        self.world = world

    def load_json_file(self) -> List[Dict]:
        """
        Load the json file.

        :return: The data from the json file.
        """
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        return data

    def load_all_objects(self):
        """
        Load the objects from the json file.
        """
        for obj in self.data:
            self.load_object(obj)

    def load_object(self, dict_obj: Dict):
        """
        Load a single object from a dictionary.

        :param dict_obj: The dictionary object to load.
        """
        name = dict_obj["class"] + "_" + str(dict_obj["id"])
        mesh_file = dict_obj["obj_file"]
        self.copy_mesh_file_to_pycram_resources(mesh_file)
        pose = self.parse_object_pose(dict_obj["pose"])
        self.pycram_objects.append(Object(name, ObjectType.ENVIRONMENT, mesh_file, pose=pose))

    def copy_mesh_file_to_pycram_resources(self, mesh_file: str):
        """
        Copy the mesh file to the PyCRAM resources folder.

        :param mesh_file: The mesh file to copy.
        """
        mesh_file_path = os.path.join(self.env_dir, mesh_file)
        shutil.copy(mesh_file_path, os.path.join(self.world.conf.resources_path, "objects"))

    @staticmethod
    def parse_object_pose(pose: Dict) -> Pose:
        """
        Parse the object pose to a PyCRAM Pose.

        :param pose: The pose to parse.
        :return: The parsed pose as a PyCRAM Pose.
        """
        rot_matrix = np.eye(4)
        rot_matrix[:3, :3] = np.array(pose["rotation"]).reshape(3, 3)
        quaternion = quaternion_from_matrix(rot_matrix)
        return Pose(pose["position"], quaternion)

