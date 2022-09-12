from copy import deepcopy
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils.constants import NoValidPositionException

class HouseBuilding3Env(BaseEnv):
  '''Open loop house building 3 task.

  The robot needs to: (1) place two cubic blocks adjacent to each other, (2) put a cuboid on
  top of the two cubic bricks, (3) put a roof on top of the cuboid.

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
    self.num_class = 2*config['num_objects'] - 1
    super(HouseBuilding3Env, self).__init__(config)

  def reset(self):
    ''''''
    while True:
      self.resetPybulletWorkspace()
      try:
        self._generateShapes(constants.ROOF, 1, random_orientation=self.random_orientation)
        self._generateShapes(constants.BRICK, 1, random_orientation=self.random_orientation)
        self._generateShapes(constants.CUBE, 2, random_orientation=self.random_orientation)
      except NoValidPositionException as e:
        continue
      else:
        break
    return self._getObservation()

  def _checkTermination(self):
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

    top_blocks = []
    for block in blocks:
      if self._isObjOnTop(block, blocks):
        top_blocks.append(block)
    if len(top_blocks) != 2:
      return False
    if self._checkOnTop(top_blocks[0], bricks[0]) and \
        self._checkOnTop(top_blocks[1], bricks[0]) and \
        self._checkOnTop(bricks[0], roofs[0]) and \
        self._checkOriSimilar([bricks[0], roofs[0]]) and \
        self._checkInBetween(bricks[0], blocks[0], blocks[1]) and \
        self._checkInBetween(roofs[0], blocks[0], blocks[1]):
      return True
    return False

  def isSimValid(self):
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
    return self._checkObjUpright(roofs[0]) and super(HouseBuilding3Env, self).isSimValid()

  def get_true_abs_state(self):
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

    if not self._checkObjUpright(roofs[0]) and BaseEnv.isSimValid(self):
      return self.num_class
    if self._checkOnTop(blocks[0], bricks[0]) and \
       self._checkOnTop(blocks[1], bricks[0]) and \
       self._checkOnTop(bricks[0], roofs[0]) and \
       self._checkInBetween(bricks[0], blocks[0], blocks[1]) and \
       self._checkInBetween(roofs[0], blocks[0], blocks[1]):
       return 0
    if self._checkOnTop(blocks[0], bricks[0]) and \
       self._checkOnTop(blocks[1], bricks[0]) and \
       self._checkInBetween(bricks[0], blocks[0], blocks[1]) and \
       self._isObjectHeld(roofs[0]):
       return 1
    if self._checkOnTop(blocks[0], bricks[0]) and \
       self._checkOnTop(blocks[1], bricks[0]) and \
       self._checkInBetween(bricks[0], blocks[0], blocks[1]) and \
       not roofs[0].isTouching(bricks[0]) and \
       not roofs[0].isTouching(blocks[0]) and \
       not roofs[0].isTouching(blocks[1]) and \
       self._isObjOnGround(roofs[0]):
       return 2

    if self._isObjectHeld(bricks[0]) and \
       self._isObjOnGround(blocks[0]) and \
       self._isObjOnGround(blocks[1]) and \
       self._isObjOnGround(roofs[0]) and \
       not roofs[0].isTouching(blocks[0]) and \
       not roofs[0].isTouching(blocks[1]) and \
       not blocks[0].isTouching(blocks[1]) and \
       not self._checkInBetween(roofs[0], blocks[0], blocks[1]) and \
       self._getDistance(blocks[0], blocks[1]) < 2.3 * self.max_block_size:
       return 3
    if self._isObjOnGround(blocks[0]) and \
       self._isObjOnGround(blocks[1]) and \
       self._isObjOnGround(bricks[0]) and \
       self._isObjOnGround(roofs[0]) and \
       not blocks[0].isTouching(blocks[1]) and \
       not blocks[0].isTouching(bricks[0]) and \
       not blocks[1].isTouching(bricks[0]) and \
       not blocks[0].isTouching(roofs[0]) and \
       not blocks[1].isTouching(roofs[0]) and \
       not bricks[0].isTouching(roofs[0]) and \
       self._getDistance(blocks[0], blocks[1]) < 2.3 * self.max_block_size and \
       not self._checkInBetween(roofs[0], blocks[0], blocks[1]) and \
       not self._checkInBetween(bricks[0], blocks[0], blocks[1]):
       return 4
    if self._isObjOnGround(blocks[0]) and \
       self._isObjOnGround(blocks[1]) and \
       self._isObjOnGround(bricks[0]) and \
       self._isObjOnGround(roofs[0]) and \
       not blocks[0].isTouching(blocks[1]) and \
       not blocks[0].isTouching(bricks[0]) and \
       not blocks[1].isTouching(bricks[0]) and \
       not blocks[0].isTouching(roofs[0]) and \
       not blocks[1].isTouching(roofs[0]) and \
       not bricks[0].isTouching(roofs[0]):
       return 6
    if self._isObjectHeld(blocks[0]) and \
       self._isObjOnGround(bricks[0]) and \
       self._isObjOnGround(blocks[1]) and \
       self._isObjOnGround(roofs[0]) and \
       not blocks[1].isTouching(bricks[0]) and \
       not blocks[1].isTouching(roofs[0]) and \
       not roofs[0].isTouching(bricks[0]):
      return 5
    if self._isObjectHeld(blocks[1]) and \
       self._isObjOnGround(bricks[0]) and \
       self._isObjOnGround(blocks[0]) and \
       self._isObjOnGround(roofs[0]) and \
       not blocks[0].isTouching(bricks[0]) and \
       not blocks[0].isTouching(roofs[0]) and \
       not roofs[0].isTouching(bricks[0]):
      return 5
    return self.num_class

def createHouseBuilding3Env(config):
  return HouseBuilding3Env(config)
