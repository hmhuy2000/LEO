import time
from copy import deepcopy
import numpy.random as npr
import numpy as np
import pybullet as pb
from itertools import combinations
from bulletarm.envs.deconstruct_envs.deconstruct_env import DeconstructEnv
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants

class HouseBuilding4DeconstructEnv(DeconstructEnv):
  ''''''
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'num_objects' not in config:
      config['num_objects'] = 6
    if 'max_steps' not in config:
      config['max_steps'] = 20
    self.num_class = 2*config['num_objects'] - 1
    super(HouseBuilding4DeconstructEnv, self).__init__(config)

  def checkStructure(self):
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

    level1_blocks = list(filter(self._isObjOnGround, blocks))
    if len(level1_blocks) != 2:
      return False

    level2_blocks = list(set(blocks) - set(level1_blocks))
    return self._checkOnTop(level1_blocks[0], bricks[0]) and \
           self._checkOnTop(level1_blocks[1], bricks[0]) and \
           self._checkOnTop(bricks[0], level2_blocks[0]) and \
           self._checkOnTop(bricks[0], level2_blocks[1]) and \
           self._checkOnTop(level2_blocks[0], roofs[0]) and \
           self._checkOnTop(level2_blocks[1], roofs[0]) and \
           self._checkOriSimilar([bricks[0], roofs[0]]) and \
           self._checkInBetween(bricks[0], level1_blocks[0], level1_blocks[1]) and \
           self._checkInBetween(roofs[0], level2_blocks[0], level2_blocks[1]) and \
           self._checkInBetween(bricks[0], level2_blocks[0], level2_blocks[1])

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
                                constants.BRICK)

    rot1 = pb.getQuaternionFromEuler([0., 0., r])
    rot2 = pb.getQuaternionFromEuler([0., 0., r])
    self.generateStructureShape((pos1[0], pos1[1], self.max_block_size * 2.5), rot1, constants.CUBE)
    self.generateStructureShape((pos2[0], pos2[1], self.max_block_size * 2.5), rot2, constants.CUBE)

    self.generateStructureShape([x, y, self.max_block_size * 3.5], pb.getQuaternionFromEuler([0., 0., r]),
                                constants.ROOF)

    self.wait(50)

  def isSimValid(self):
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
    return self._checkObjUpright(roofs[0]) and super(HouseBuilding4DeconstructEnv, self).isSimValid() 

  def check_block_touching(self, blocks):
    for i in range(4):
      for j in range(i+1, 4):
        if blocks[i].isTouching(blocks[j]):
          return True
        if self._checkOnTop(blocks[i], blocks[j]) or self._checkOnTop(blocks[j], blocks[i]):
          return True
    return False

  def get_true_abs_state(self):
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

    level1_blocks = list(filter(self._isObjOnGround, blocks))
    level2_blocks = list(set(blocks) - set(level1_blocks))

    if not self._checkObjUpright(roofs[0]) and BaseEnv.isSimValid(self):
      return self.num_class
    if len(level1_blocks) == 2 and len(level2_blocks) == 2:
      if self._checkOnTop(level1_blocks[0], bricks[0]) and \
             self._checkOnTop(level1_blocks[1], bricks[0]) and \
             self._checkOnTop(bricks[0], level2_blocks[0]) and \
             self._checkOnTop(bricks[0], level2_blocks[1]) and \
             self._checkOnTop(level2_blocks[0], roofs[0]) and \
             self._checkOnTop(level2_blocks[1], roofs[0]) and \
             self._checkOriSimilar([bricks[0], roofs[0]]) and \
             self._checkInBetween(bricks[0], level1_blocks[0], level1_blocks[1]) and \
             self._checkInBetween(roofs[0], level2_blocks[0], level2_blocks[1]) and \
             self._checkInBetween(bricks[0], level2_blocks[0], level2_blocks[1]):
             return 0
      if self._checkOnTop(level1_blocks[0], bricks[0]) and \
             self._checkOnTop(level1_blocks[1], bricks[0]) and \
             self._checkOnTop(bricks[0], level2_blocks[0]) and \
             self._checkOnTop(bricks[0], level2_blocks[1]) and \
             self._isObjectHeld(roofs[0]) and \
             self._checkInBetween(bricks[0], level1_blocks[0], level1_blocks[1]) and \
             self._checkInBetween(bricks[0], level2_blocks[0], level2_blocks[1]):
             return 1
      if self._checkOnTop(level1_blocks[0], bricks[0]) and \
             self._checkOnTop(level1_blocks[1], bricks[0]) and \
             self._checkOnTop(bricks[0], level2_blocks[0]) and \
             self._checkOnTop(bricks[0], level2_blocks[1]) and \
             self._isObjOnGround(roofs[0]) and \
             self._checkInBetween(bricks[0], level1_blocks[0], level1_blocks[1]) and \
             self._checkInBetween(bricks[0], level2_blocks[0], level2_blocks[1]):
             return 2
    
      for i in [0, 1]:
        if self._checkOnTop(level1_blocks[0], bricks[0]) and \
             self._checkOnTop(level1_blocks[1], bricks[0]) and \
             self._checkOnTop(bricks[0], level2_blocks[i]) and \
             self._isObjectHeld(level2_blocks[1-i]) and \
             self._checkInBetween(bricks[0], level1_blocks[0], level1_blocks[1]) and \
             self._isObjOnGround(roofs[0]):
             return 3
    elif len(level1_blocks) == 3:
      for i in range(3):
        for j in range(3):
          if i != j:
            if self._checkOnTop(level1_blocks[i], bricks[0]) and \
               self._checkOnTop(level1_blocks[j], bricks[0]) and \
               self._checkOnTop(bricks[0], level2_blocks[0]) and \
               self._checkInBetween(bricks[0], level1_blocks[i], level1_blocks[j]) and \
               self._isObjOnGround(roofs[0]) and \
               self._isObjOnGround(level1_blocks[3-i-j]) and \
               not level1_blocks[3-i-j].isTouching(roofs[0]):
               return 4
            if self._checkOnTop(level1_blocks[i], bricks[0]) and \
               self._checkOnTop(level1_blocks[j], bricks[0]) and \
               self._checkInBetween(bricks[0], level1_blocks[i], level1_blocks[j]) and \
               self._isObjOnGround(roofs[0]) and \
               self._isObjectHeld(level2_blocks[0]) and \
               self._isObjOnGround(level1_blocks[3-i-j]) and \
               not level1_blocks[3-i-j].isTouching(roofs[0]):
               return 5
            if self._isObjOnGround(bricks[0]) and \
               self._isObjOnGround(roofs[0]) and \
               self._isObjectHeld(level2_blocks[0]) and \
               self._isObjOnGround(level1_blocks[i]) and \
               self._isObjOnGround(level1_blocks[j]) and \
               self._isObjOnGround(level1_blocks[3-i-j]) and \
               not roofs[0].isTouching(level1_blocks[i]) and \
               not roofs[0].isTouching(level1_blocks[j]) and \
               not roofs[0].isTouching(level1_blocks[3-i-j]) and \
               not bricks[0].isTouching(level1_blocks[i]) and \
               not bricks[0].isTouching(level1_blocks[j]) and \
               not bricks[0].isTouching(level1_blocks[3-i-j]) and \
               not level1_blocks[i].isTouching(level1_blocks[j]) and \
               not level1_blocks[i].isTouching(level1_blocks[3-i-j]) and \
               not roofs[0].isTouching(bricks[0]):
               return 9
    else:
      for i in range(4):
        for j in range(4):
          for k in range(4):
            if i != j and i != k and k != j:
              if self._checkOnTop(blocks[i], bricks[0]) and \
                   self._checkOnTop(blocks[j], bricks[0]) and \
                   self._checkInBetween(bricks[0], blocks[i], blocks[j]) and \
                   self._isObjOnGround(roofs[0]) and \
                   self._isObjOnGround(blocks[k]) and \
                   self._isObjOnGround(blocks[6-i-j-k]) and \
                   not blocks[k].isTouching(roofs[0]) and \
                   not blocks[6-i-j-k].isTouching(roofs[0]) and \
                   not blocks[k].isTouching(blocks[6-i-j-k]):
                   return 6
              if self._getDistance(blocks[i], blocks[j]) < 2.4 * self.max_block_size and \
                   self._isObjectHeld(bricks[0]) and \
                   self._isObjOnGround(roofs[0]) and \
                   self._isObjOnGround(blocks[i]) and \
                   self._isObjOnGround(blocks[j]) and \
                   self._isObjOnGround(blocks[k]) and \
                   self._isObjOnGround(blocks[6-i-j-k]) and \
                   not self._checkInBetween(roofs[0], blocks[i], blocks[j]) and \
                   not self._checkInBetween(blocks[k], blocks[i], blocks[j]) and \
                   not self._checkInBetween(blocks[6-i-j-k], blocks[i], blocks[j]) and \
                   not roofs[0].isTouching(blocks[i]) and \
                   not roofs[0].isTouching(blocks[j]) and \
                   not roofs[0].isTouching(blocks[k]) and \
                   not roofs[0].isTouching(blocks[6-i-j-k]) and \
                   not self.check_block_touching(blocks):
                   return 7
              if self._getDistance(blocks[i], blocks[j]) < 2.3 * self.max_block_size and \
                   self._isObjOnGround(bricks[0]) and \
                   self._isObjOnGround(roofs[0]) and \
                   self._isObjOnGround(blocks[i]) and \
                   self._isObjOnGround(blocks[j]) and \
                   self._isObjOnGround(blocks[k]) and \
                   self._isObjOnGround(blocks[6-i-j-k]) and \
                   not self._checkInBetween(roofs[0], blocks[i], blocks[j]) and \
                   not self._checkInBetween(bricks[0], blocks[i], blocks[j]) and \
                   not self._checkInBetween(blocks[k], blocks[i], blocks[j]) and \
                   not self._checkInBetween(blocks[6-i-j-k], blocks[i], blocks[j]) and \
                   not roofs[0].isTouching(blocks[i]) and \
                   not roofs[0].isTouching(blocks[j]) and \
                   not roofs[0].isTouching(blocks[k]) and \
                   not roofs[0].isTouching(blocks[6-i-j-k]) and \
                   not bricks[0].isTouching(blocks[i]) and \
                   not bricks[0].isTouching(blocks[j]) and \
                   not bricks[0].isTouching(blocks[k]) and \
                   not bricks[0].isTouching(blocks[6-i-j-k]) and \
                   not roofs[0].isTouching(bricks[0]) and \
                   not self.check_block_touching(blocks):
                   return 8

      for i in range(4):
        for j in range(4):
          for k in range(4):      
              if i !=j and j!=k and k != i and \
                   self._isObjOnGround(bricks[0]) and \
                   self._isObjOnGround(roofs[0]) and \
                   self._isObjOnGround(blocks[i]) and \
                   self._isObjOnGround(blocks[j]) and \
                   self._isObjOnGround(blocks[k]) and \
                   self._isObjOnGround(blocks[6-i-j-k]) and \
                   not roofs[0].isTouching(blocks[i]) and \
                   not roofs[0].isTouching(blocks[j]) and \
                   not roofs[0].isTouching(blocks[k]) and \
                   not roofs[0].isTouching(blocks[6-i-j-k]) and \
                   not bricks[0].isTouching(blocks[i]) and \
                   not bricks[0].isTouching(blocks[j]) and \
                   not bricks[0].isTouching(blocks[k]) and \
                   not bricks[0].isTouching(blocks[6-i-j-k]) and \
                   not roofs[0].isTouching(bricks[0]) and \
                   not roofs[0].isTouching(bricks[0]) and \
                   not self.check_block_touching(blocks):
                   return 10
    return self.num_class  

def createHouseBuilding4DeconstructEnv(config):
  return HouseBuilding4DeconstructEnv(config)
