import time
from copy import deepcopy
import numpy.random as npr
import numpy as np
from itertools import combinations
from bulletarm.envs.deconstruct_envs.deconstruct_env import DeconstructEnv
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants

class BlockStackingDeconstructEnv(DeconstructEnv):
  ''''''
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'num_objects' not in config:
      config['num_objects'] = 4
    if 'max_steps' not in config:
      config['max_steps'] = 10
    self.num_class = config['num_objects'] * 2 - 1
    super(BlockStackingDeconstructEnv, self).__init__(config)

  def checkStructure(self):
    ''''''
    return self._checkStack()

  def generateStructure(self):
    padding = self.max_block_size * 1.5
    pos = self.get1BaseXY(padding)
    rot = self._getValidOrientation(self.random_orientation)
    for i in range(self.num_obj):
      self.generateStructureShape((pos[0], pos[1], i * self.max_block_size + self.max_block_size / 2), rot,
                                  constants.CUBE)
    self.wait(50)
    
  def get_true_abs_state(self):
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    if not BaseEnv.isSimValid(self):
      return self.num_class
    if self._checkStack():
      return 0
    
    for i in range(4):
      for j in range(4):
        for k in range(4):
          if i != j and j != k and i != k:
            if self._checkStack([blocks[i], blocks[j], blocks[k]]) and \
                self._isObjectHeld(blocks[6-i-j-k]):
              return 1
            if self._checkStack([blocks[i], blocks[j], blocks[k]]) and \
                self._isObjOnGround(blocks[6-i-j-k]) and \
                not blocks[6-i-j-k].isTouching(blocks[j]) and \
                not blocks[6-i-j-k].isTouching(blocks[j]) and \
                not blocks[6-i-j-k].isTouching(blocks[k]):                
              return 2
            if self._checkStack([blocks[i], blocks[j]]) and \
                self._isObjectHeld(blocks[k]) and \
                self._isObjOnGround(blocks[6-i-j-k]) and \
                not blocks[6-i-j-k].isTouching(blocks[i]) and \
                not blocks[6-i-j-k].isTouching(blocks[j]):
              return 3
            if self._checkStack([blocks[i], blocks[j]]) and \
                self._isObjOnGround(blocks[k]) and \
                self._isObjOnGround(blocks[6-i-j-k]) and \
                not blocks[6-i-j-k].isTouching(blocks[i]) and \
                not blocks[6-i-j-k].isTouching(blocks[j]) and \
                not blocks[k].isTouching(blocks[i]) and \
                not blocks[k].isTouching(blocks[i]) and \
                not blocks[6-i-j-k].isTouching(blocks[k]):
              return 4
            if self._isObjOnGround(blocks[i]) and \
                self._isObjOnGround(blocks[k]) and \
                self._isObjOnGround(blocks[6-i-j-k]) and \
                self._isObjectHeld(blocks[j]) and \
                not blocks[6-i-j-k].isTouching(blocks[k]) and \
                not blocks[6-i-j-k].isTouching(blocks[i]) and \
                not blocks[i].isTouching(blocks[k]):
              return 5
            if self._isObjOnGround(blocks[i]) and \
                self._isObjOnGround(blocks[j]) and \
                self._isObjOnGround(blocks[k]) and \
                self._isObjOnGround(blocks[6-i-j-k]) and \
                not blocks[6-i-j-k].isTouching(blocks[k]) and \
                not blocks[6-i-j-k].isTouching(blocks[j]) and \
                not blocks[6-i-j-k].isTouching(blocks[i]) and \
                not blocks[i].isTouching(blocks[j]) and \
                not blocks[i].isTouching(blocks[k]) and \
                not blocks[j].isTouching(blocks[k]):
              return 6 
    return self.num_class


def createBlockStackingDeconstructEnv(config):
  return BlockStackingDeconstructEnv(config)
