from apel.apel import APEL
from pycram.datastructures.enums import WorldMode, ObjectType
from pycram.world_concepts.world_object import Object
from pycram.worlds.bullet_world import BulletWorld


world = BulletWorld(mode=WorldMode.GUI)
apel = APEL("../envs/scene_bundle_1/scene.json", world.current_world)

apel.load_all_objects()
pr2 = Object("pr2", ObjectType.ROBOT, path="pr2.urdf")
