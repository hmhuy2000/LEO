import unittest
import time
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from bulletarm_baselines.fc_dqn.utils.parameters import *

import matplotlib.pyplot as plt

from bulletarm import env_factory




class TestBulletBlockStacking(unittest.TestCase):
  env_config = {'num_objects': 4}

  planner_config = {'random_orientation': True}

  def testPlanner(self):
    self.env_config['render'] = True
    env = env_factory.createEnvs(1, 'block_stacking', self.env_config, self.planner_config)
    (states_, in_hands_, obs_) = env.reset()
    print(states_,in_hands_.shape,obs_.shape)
    abs_state = (4)*2 - 2 
    for i in range(5, -1, -1):
      action = env.getNextAction()
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
      plt.imshow(obs_.reshape((128,128)))
      plt.show()
      plt.imshow(in_hands_.reshape((24,24)))
      plt.show()
      abs_state -= 1
    print(f'success at state {abs_state}')
    env.close()

  def testPlanner2(self):
    self.env_config['render'] = False
    self.env_config['seed'] = 0
    num_processes = 20
    env = env_factory.createEnvs(num_processes,  'block_stacking', self.env_config, self.planner_config)
    total = 0
    s = 0
    step_times = []
    env.reset()
    pbar = tqdm(total=500)
    while total < 500:
      t0 = time.time()
      action = env.getNextAction()
      t_plan = time.time() - t0
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=True)
      s += rewards.sum()
      total += dones.sum()
      t_action = time.time() - t0 - t_plan
      t = time.time() - t0
      step_times.append(t)

      pbar.set_description(
        '{}/{}, SR: {:.3f}, plan time: {:.2f}, action time: {:.2f}, avg step time: {:.2f}'
          .format(s, total, float(s) / total if total != 0 else 0, t_plan, t_action, np.mean(step_times))
      )
      pbar.update(dones.sum())
    env.close()

if __name__ == '__main__':
  env = TestBulletBlockStacking()
  env.testPlanner()