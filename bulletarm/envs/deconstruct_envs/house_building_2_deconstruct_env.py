from ast import Constant
import dis
import time
from copy import deepcopy
import numpy.random as npr
import numpy as np
import pybullet as pb
from itertools import combinations
from bulletarm.envs.deconstruct_envs.deconstruct_env import DeconstructEnv
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants

class HouseBuilding2DeconstructEnv(DeconstructEnv):
  ''''''
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'num_objects' not in config:
      config['num_objects'] = 3
    if 'max_steps' not in config:
      config['max_steps'] = 10
    self.num_class = config['num_objects']*2-1
    super(HouseBuilding2DeconstructEnv, self).__init__(config)

  def checkStructure(self):
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

    return self._checkOnTop(blocks[0], roofs[0]) and \
           self._checkOnTop(blocks[1], roofs[0]) and \
           self._checkInBetween(roofs[0], blocks[0], blocks[1])

  def generateStructure(self):
    padding = self.max_block_size * 1.5
    min_dist = 2.1 * self.max_block_size
    max_dist = 2.2 * self.max_block_size
    pos1, pos2 = self.get2BaseXY(padding, min_dist, max_dist)
    rot1 = self._getValidOrientation(self.random_orientation)
    rot2 = self._getValidOrientation(self.random_orientation)
    self.generateStructureShape((pos1[0], pos1[1], self.max_block_size / 2), rot1, constants.CUBE)
    self.generateStructureShape((pos2[0], pos2[1], self.max_block_size / 2), rot2, constants.CUBE)

    x, y, r = self.getXYRFrom2BasePos(pos1, pos2)
    self.generateStructureShape([x, y, self.max_block_size * 1.5], pb.getQuaternionFromEuler([0., 0., r]),
                                constants.ROOF)
    self.wait(50)

  def isSimValid(self):
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
    return self._checkObjUpright(roofs[0]) and super(HouseBuilding2DeconstructEnv, self).isSimValid()

  def get_true_abs_state(self):
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
    
    if not self._checkObjUpright(roofs[0]) or not BaseEnv.isSimValid(self):
      return self.num_class
    dist_blocks = self._getDistance(blocks[0], blocks[1]) 
    if self._checkOnTop(blocks[0], roofs[0]) and \
       self._checkOnTop(blocks[1], roofs[0]) and \
       self._checkInBetween(roofs[0], blocks[0], blocks[1]):
      return 0
    if self._isObjectHeld(roofs[0]) and \
       self._isObjOnGround(blocks[0]) and \
       self._isObjOnGround(blocks[1]) and \
       dist_blocks < 3.0 * self.max_block_size:
      return 1
    if self._isObjOnGround(blocks[0]) and \
       self._isObjOnGround(blocks[1]) and \
       self._isObjOnGround(roofs[0]) and \
       not roofs[0].isTouching(blocks[0]) and \
       not roofs[0].isTouching(blocks[1]) and \
       not self._checkInBetween(roofs[0], blocks[0], blocks[1]) and \
       dist_blocks < 3.0 * self.max_block_size:
      return 2
    if self._isObjOnGround(blocks[0]) and \
       self._isObjOnGround(blocks[1]) and \
       self._isObjOnGround(roofs[0]) and \
       not roofs[0].isTouching(blocks[0]) and \
       not roofs[0].isTouching(blocks[1]):
      return 4
    if self._isObjectHeld(blocks[0]) and \
       self._isObjOnGround(blocks[1]) and \
       self._isObjOnGround(roofs[0]) and \
       not roofs[0].isTouching(blocks[1]):
      return 3
    if self._isObjectHeld(blocks[1]) and \
       self._isObjOnGround(blocks[0]) and \
       self._isObjOnGround(roofs[0]) and \
       not roofs[0].isTouching(blocks[0]):
      return 3      
    return self.num_class

def createHouseBuilding2DeconstructEnv(config):
  return HouseBuilding2DeconstructEnv(config)
