import unittest
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from bulletarm import env_factory

class TestBulletHouse2(unittest.TestCase):
  env_config = {'random_orientation': False}
  planner_config = {'pos_noise': 0, 'rot_noise': 0}

  def testStepLeft(self):
    self.env_config['render'] = True
    env = env_factory.createEnvs(1,  'house_building_2', self.env_config, self.planner_config)
    env.reset()

    positions = env.getObjectPositions()[0]
    # pick up the roof
    action = [[0, positions[2][0], positions[2][1], 0]]
    (states_, in_hands_, obs_), rewards, dones = env.step(np.array(action), auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 5)
    self.assertEqual(dones, 0)

    (states_, in_hands_, obs_), rewards, dones = env.step(env.getNextAction(), auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 4)
    self.assertEqual(dones, 0)

    positions = env.getObjectPositions()[0]
    action = [[0, positions[1][0], positions[1][1], 0]]
    (states_, in_hands_, obs_), rewards, dones = env.step(np.array(action), auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 3)
    self.assertEqual(dones, 0)

    (states_, in_hands_, obs_), rewards, dones = env.step(env.getNextAction(), auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 2)
    self.assertEqual(dones, 0)

    (states_, in_hands_, obs_), rewards, dones = env.step(env.getNextAction(), auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 1)
    self.assertEqual(dones, 0)

    (states_, in_hands_, obs_), rewards, dones = env.step(env.getNextAction(), auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 0)
    self.assertEqual(dones, 1)

    env.close()

  def testPlanner(self):
    self.env_config['render'] = True
    env = env_factory.createEnvs(1,  'house_building_2', self.env_config, self.planner_config)
    (states_, in_hands_, obs_) = env.reset()
    print(states_,in_hands_.shape,obs_.shape)
    abs_state = (3)*2 - 2 
    plt.imshow(obs_.reshape((128,128)))
    plt.tight_layout()
    plt.savefig(f'images/deep_{abs_state}.png')
    plt.show()
    plt.imshow(in_hands_.reshape((24,24)))
    plt.tight_layout()
    plt.savefig(f'images/inhand_{abs_state}.png')
    plt.show()
    while(abs_state):
      action = env.getNextAction()
      print(abs_state)
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
      abs_state -= 1
      plt.imshow(obs_.reshape((128,128)))
      plt.tight_layout()
      plt.savefig(f'images/deep_{abs_state}.png')
      plt.show()
      plt.imshow(in_hands_.reshape((24,24)))
      plt.tight_layout()
      plt.savefig(f'images/inhand_{abs_state}.png')
      plt.show()
      if (dones):
        break
    print(f'success at state {abs_state}')
    env.close()

if __name__ == '__main__':
  env = TestBulletHouse2()
  env.testPlanner()