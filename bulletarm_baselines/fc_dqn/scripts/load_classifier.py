from bulletarm_baselines.fc_dqn.utils.SoftmaxClassifier import SoftmaxClassifier
from bulletarm_baselines.fc_dqn.utils.View import View
from bulletarm_baselines.fc_dqn.utils.ConvEncoder import ConvEncoder
from bulletarm_baselines.fc_dqn.utils.SplitConcat import SplitConcat
from bulletarm_baselines.fc_dqn.utils.FCEncoder import FCEncoder
from bulletarm_baselines.fc_dqn.utils.dataset import ArrayDataset, count_objects
from bulletarm_baselines.fc_dqn.utils.result import Result
import torch
import torch.nn as nn
from bulletarm_baselines.fc_dqn.utils.parameters import *

class block_stacking_perfect_classifier(nn.Module):
  def __init__(self):

    super(block_stacking_perfect_classifier, self).__init__()
  
  def check_equal(self, a ,b):
    return abs(a-b)<0.001

  def forward(self,obs,inhand):
    len = obs.shape[0]
    res = []
    for i in range(len):
        obs_height = torch.max(obs[i])
        in_hand_height = torch.max(inhand[i])
        if (not (self.check_equal(in_hand_height,0) or self.check_equal(in_hand_height,0.03))):
            in_hand_height = torch.tensor(0.03)

        if (self.check_equal(obs_height,0.03) and self.check_equal(in_hand_height,0)):
            res.append(6)
            continue
        if (self.check_equal(obs_height,0.03) and self.check_equal(in_hand_height,0.03)):
            res.append(5)
            continue
        if (self.check_equal(obs_height,0.06) and self.check_equal(in_hand_height,0)):
            res.append(4)
            continue
        if (self.check_equal(obs_height,0.06) and self.check_equal(in_hand_height,0.03)):
            res.append(3)
            continue
        if (self.check_equal(obs_height,0.09) and self.check_equal(in_hand_height,0)):
            res.append(2)
            continue
        if (self.check_equal(obs_height,0.09) and self.check_equal(in_hand_height,0.03)):
            res.append(1)
            continue
        if (self.check_equal(obs_height,0.12) and self.check_equal(in_hand_height,0)):
            res.append(0)
            continue
        res.append(6)
        # raise NotImplementedError(f'error classifier with obs_height = {obs_height}, in_hand_height = {in_hand_height}')
        
    return torch.tensor(res).to('cuda')

