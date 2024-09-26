import os
from unittest import TestCase
from apel.apel import APEL
from pycram import World
from pycram.datastructures.enums import WorldMode
from pycram.datastructures.pose import Pose
from pycram.ros.viz_marker_publisher import VizMarkerPublisher
from pycram.world_concepts.world_object import Object
from pycram.worlds.bullet_world import BulletWorld


class TestAPEL(TestCase):
    apel: APEL
    world: World
    viz_marker_publisher: VizMarkerPublisher

    @classmethod
    def setUpClass(cls):
        cls.world = BulletWorld(mode=WorldMode.DIRECT)
        cls.apel = APEL("../envs/scene_bundle_1/scene.json", cls.world.current_world)
        cls.viz_marker_publisher = VizMarkerPublisher()

    @classmethod
    def tearDownClass(cls):
        cls.viz_marker_publisher._stop_publishing()
        cls.world.exit()

    def tearDown(self):
        pass

    def test_load_json_file(self):
        self.assertTrue(isinstance(self.apel.data, list))
        self.assertTrue(isinstance(self.apel.data[0], dict))

    def test_parse_object_pose(self):
        pose = self.apel.parse_object_pose(self.apel.data[0]["pose"])
        self.assertTrue(isinstance(pose, Pose))

    def test_copy_mesh_file_to_pycram_resources(self):
        self.apel.copy_mesh_file_to_pycram_resources(self.apel.data[0]["obj_file"])
        self.assertTrue(os.path.exists(os.path.join(self.world.conf.resources_path, "objects",
                                                    self.apel.data[0]["obj_file"])))

    def test_load_object(self):
        self.apel.load_generic_object(self.apel.data[0])
        self.assertTrue(len(self.apel.pycram_objects) > 0)
        self.assertTrue(isinstance(self.apel.pycram_objects[0], Object))

    def test_load_all_objects(self):
        self.apel.load_all_objects()
        self.assertTrue(len(self.apel.pycram_objects) == len(self.apel.data))
        self.assertTrue(all([isinstance(obj, Object) for obj in self.apel.pycram_objects]))

