from copy import deepcopy
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils.constants import NoValidPositionException

class HouseBuilding1Env(BaseEnv):
  '''Open loop house building 1 task.

  The robot needs to first stack N-1 cubic blocks then place a triangle block on top of
  the stack. The number of blocks N is set by the config.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'num_objects' not in config:
      config['num_objects'] = 4
    if 'max_steps' not in config:
      config['max_steps'] = 10
    self.num_class = config['num_objects'] * 2 - 1
    super(HouseBuilding1Env, self).__init__(config)

  def reset(self):
    ''''''
    while True:
      self.resetPybulletWorkspace()
      try:
        self._generateShapes(constants.TRIANGLE, 1, random_orientation=self.random_orientation)
        self._generateShapes(constants.CUBE, self.num_obj-1, random_orientation=self.random_orientation)
      except NoValidPositionException as e:
        continue
      else:
        break
    return self._getObservation()

  def _checkTermination(self):
    ''''''
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
    return self._checkStack(blocks+triangles) and self._checkObjUpright(triangles[0])

  def isSimValid(self):
    triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
    return self._checkObjUpright(triangles[0]) and super(HouseBuilding1Env, self).isSimValid()

  def get_other(self,i,j):
    if (i+j == 3):
      return 0
    if (i+j == 2):
      return 1
    if (i+j == 1):
      return 2

  def get_true_abs_state(self):
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
    
    if not self._checkObjUpright(triangles[0]) or not BaseEnv.isSimValid(self):
      return self.num_class
    if (self._checkStack(blocks+triangles) and self._checkObjUpright(triangles[0])):
      return 0
    if (self._checkStack(blocks) and (self._isObjectHeld(triangles[0]))):
      return 1
    if (self._checkStack(blocks) and (not self._isObjectHeld(triangles[0]))):
      return 2
    for i in range(3):
      for j in range(3):
        if i != j:
          other = self.get_other(i,j)
          if self._checkOnTop(blocks[i], blocks[j]) and \
              self._isObjectHeld(blocks[other]) and \
              self._isObjOnGround(triangles[0]) and \
              self._isObjOnGround(blocks[i]) and \
              not triangles[0].isTouching(blocks[i]) and \
              not triangles[0].isTouching(blocks[j]):
              return 3

          if self._checkOnTop(blocks[i], blocks[j]) and \
              self._isObjOnGround(blocks[other]) and \
              self._isObjOnGround(blocks[i]) and \
              self._isObjOnGround(triangles[0]) and \
              not triangles[0].isTouching(blocks[i]) and \
              not triangles[0].isTouching(blocks[j]) and \
              not triangles[0].isTouching(blocks[other]) and \
              not blocks[other].isTouching(blocks[i]) and \
              not blocks[other].isTouching(blocks[j]):
              return 4

          if self._isObjectHeld(blocks[j]) and \
              self._isObjOnGround(blocks[i]) and \
              self._isObjOnGround(blocks[other]) and \
              self._isObjOnGround(triangles[0]) and \
              not triangles[0].isTouching(blocks[i]) and \
              not triangles[0].isTouching(blocks[other]) and \
              not blocks[other].isTouching(blocks[i]):
              return 5

          if self._isObjOnGround(blocks[i]) and \
              self._isObjOnGround(blocks[j]) and \
              self._isObjOnGround(blocks[other]) and \
              self._isObjOnGround(triangles[0]) and \
              not triangles[0].isTouching(blocks[i]) and \
              not triangles[0].isTouching(blocks[j]) and \
              not triangles[0].isTouching(blocks[other]) and \
              not blocks[other].isTouching(blocks[i]) and \
              not blocks[other].isTouching(blocks[j]) and \
              not blocks[i].isTouching(blocks[j]):
              return 6
    return self.num_class

def createHouseBuilding1Env(config):
  return HouseBuilding1Env(config)
