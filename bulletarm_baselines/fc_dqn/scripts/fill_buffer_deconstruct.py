from asyncio.log import logger
import sys
import time
import copy
import collections
import torch
from datetime import datetime
from tqdm import tqdm
from bulletarm_baselines.fc_dqn.utils.parameters import *
from bulletarm_baselines.fc_dqn.utils.dataset import ListDataset
import matplotlib.pyplot as plt
import os
import argparse

sys.path.append('./')
sys.path.append('..')
from bulletarm_baselines.fc_dqn.utils.env_wrapper import EnvWrapper

def create_folder(path):
    try:
        os.mkdir(path)
    except:
        print(f'[INFO] folder {path} existed, can not create new')

ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert abs_state abs_goal abs_state_next abs_goal_next')


def update_abs_goals(abs_states):
    with torch.no_grad():
        zeros_goals = torch.zeros_like(abs_states)
        return torch.max(abs_states - 1, zeros_goals)

def getCurrentObs(in_hand, obs):
    obss = []
    for i, o in enumerate(obs):
        obss.append((o.squeeze(), in_hand[i].squeeze()))
    return obss

def fillDeconstruct(agent, replay_buffer):
    def states_valid(states_list):
        if len(states_list) < 2:
            return False
        for i in range(1, len(states_list)):
            if states_list[i] != 1 - states_list[i-1]:
                return False
        return True

    def rewards_valid(reward_list):
        if reward_list[0] != 1:
            return False
        for i in range(1, len(reward_list)):
            if reward_list[i] != 0:
                return False
        return True

    if env in ['block_stacking',
               'house_building_1',
               'house_building_2',
               'house_building_3',
               'house_building_4',
               'improvise_house_building_2',
               'improvise_house_building_3',
               'improvise_house_building_discrete',
               'improvise_house_building_random',
               'ramp_block_stacking',
               'ramp_house_building_1',
               'ramp_house_building_2',
               'ramp_house_building_3',
               'ramp_house_building_4',
               'ramp_improvise_house_building_2',
               'ramp_improvise_house_building_3']:
        deconstruct_env = env + '_deconstruct'
    else:
        raise NotImplementedError('deconstruct env not supported for env: {}'.format(env))

    plt.style.use('default')
    envs = EnvWrapper(num_processes, deconstruct_env, env_config, planner_config)

    states, in_hands, obs = envs.reset()
    total = 0
    s = 0
    step_times = []
    steps = [0 for i in range(num_processes)]
    local_state = [[] for i in range(num_processes)]
    local_obs = [[] for i in range(num_processes)]
    local_action = [[] for i in range(num_processes)]
    local_reward = [[] for i in range(num_processes)]

    pbar = tqdm(total=planner_episode)
    while total < planner_episode:
        # buffer_obs = agent.getCurrentObs(in_hands, obs)
        plan_actions = envs.getNextAction()
        actions_star_idx, actions_star = agent.getActionFromPlan(plan_actions)
        actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
        t0 = time.time()
        states_, in_hands_, obs_, rewards, dones = envs.step(actions_star, auto_reset=False)
        state_id = action_sequence.find('p')
        dones[actions_star[:, state_id] + states_ != 1] = 1
        t = time.time()-t0
        step_times.append(t)

        buffer_obs = getCurrentObs(in_hands_, obs)
        for i in range(num_processes):
            local_state[i].append(states[i])
            local_obs[i].append(buffer_obs[i])
            local_action[i].append(actions_star_idx[i])
            local_reward[i].append(rewards[i])

        steps = list(map(lambda x: x + 1, steps))

        done_idxes = torch.nonzero(dones).squeeze(1)
        if done_idxes.shape[0] != 0:
            empty_in_hands = envs.getEmptyInHand()

            buffer_obs_ = getCurrentObs(empty_in_hands, copy.deepcopy(obs_))
            reset_states_, reset_in_hands_, reset_obs_ = envs.reset_envs(done_idxes)
            for i, idx in enumerate(done_idxes):
                local_obs[idx].append(buffer_obs_[idx])
                local_state[idx].append(copy.deepcopy(states_[idx]))
                if (num_objects-2)*2 <= steps[idx] <= num_objects*2 and states_valid(local_state[idx]) and rewards_valid(local_reward[idx]):
                    s += 1
                    for j in range(len(local_reward[idx])):
                        obs = local_obs[idx][j+1]
                        next_obs = local_obs[idx][j]

                        replay_buffer.add(ExpertTransition(local_state[idx][j+1],
                                                           obs,
                                                           local_action[idx][j],
                                                           local_reward[idx][j],
                                                           local_state[idx][j],
                                                           next_obs,
                                                           torch.tensor(float(j == 0)),
                                                           torch.tensor(float(j)),
                                                           torch.tensor(1)))

                states_[idx] = reset_states_[i]
                obs_[idx] = reset_obs_[i]

                total += 1
                steps[idx] = 0
                local_state[idx] = []
                local_obs[idx] = []
                local_action[idx] = []
                local_reward[idx] = []

        pbar.set_description(
            '{}/{}, SR: {:.3f}, step time: {:.2f}; avg step time: {:.2f}'
            .format(s, total, float(s)/total if total !=0 else 0, t, np.mean(step_times))
        )
        pbar.update(done_idxes.shape[0])

        states = copy.copy(states_)
        obs = copy.copy(obs_)
    pbar.close()
    envs.close()

def get_cls(classifier, obs, inhand):
    obs = torch.tensor(obs).type(torch.cuda.FloatTensor).to(device)
    inhand = torch.tensor(inhand).type(torch.cuda.FloatTensor).to(device)
    res = classifier([obs,inhand])
    return torch.argmax(res,dim=1)

def remove_outlier(abs_states,num_classes):
    with torch.no_grad():
        max_abs = torch.full(abs_states.shape,num_classes - 1,dtype=abs_states.dtype).to(device)
        return torch.min(abs_states, max_abs)

def train_fillDeconstructUsingRunner(agent, replay_buffer):
  if env in ['block_stacking',
             'house_building_1',
             'house_building_2',
             'house_building_3',
             'house_building_4',
             'improvise_house_building_2',
             'improvise_house_building_3',
             'improvise_house_building_discrete',
             'improvise_house_building_random',
             'ramp_block_stacking',
             'ramp_house_building_1',
             'ramp_house_building_2',
             'ramp_house_building_3',
             'ramp_house_building_4',
             'ramp_improvise_house_building_2',
             'ramp_improvise_house_building_3']:
    deconstruct_env = env + '_deconstruct'
  elif env in ['1l1b1r', '1l2b2r', '1l1l1r', '1l1l2r', '1l2b1r', '1l2b2b2r', '1l2b1l2b2r']:
        deconstruct_env = 'house_building_x' + '_deconstruct'
        env_config['goal_string'] = env
  else:
    raise NotImplementedError('deconstruct env not supported for env: {}'.format(env))
#   env_config['render'] = True
  decon_envs = EnvWrapper(num_processes, deconstruct_env, env_config, planner_config)
  num_class = 2 * decon_envs.getNumObj() - 1
  print(f'num_class in deconstruct env: {num_class}')
  cnt = 0
  transitions = decon_envs.gatherDeconstructTransitions(planner_episode)
  for i, transition in enumerate(transitions):
    (state, in_hand, obs), action, reward, done, (next_state, next_in_hand, next_obs),(abs_state,abs_state_next) = transition
    true_abs_state = torch.tensor(abs_state).to(device)

    abs_state = (i+1) % (num_class - 1)
    if (abs_state == 0):
        abs_state = num_class - 1
    abs_state = torch.tensor(abs_state).to(device)
    abs_state = remove_outlier(abs_state, num_class)
    # assert abs_state == true_abs_state
    
    true_abs_state_next = torch.tensor(abs_state_next).to(device)
    abs_state_next = abs_state - 1
    # assert abs_state_next == true_abs_state_next
    abs_state_next = remove_outlier(abs_state_next,num_class)

    abs_goal = update_abs_goals(abs_state)
    abs_goal_next =  update_abs_goals(abs_state_next)
    
    actions_star_idx, actions_star = agent.getActionFromPlan(torch.tensor(np.expand_dims(action, 0)))
    replay_buffer.add(ExpertTransition(
      torch.tensor(state).float(),
      (torch.tensor(obs).float(), torch.tensor(in_hand).float()),
      actions_star_idx[0],
      #------------------#
      torch.tensor(reward).float(),
    #   torch.tensor(0.0).float(),
      #------------------#
      torch.tensor(next_state).float(),
      (torch.tensor(next_obs).float(), torch.tensor(next_in_hand).float()),
      #------------------#
    #   torch.tensor(float(done)),
      torch.tensor(float(1)),
      #------------------#
      torch.tensor(float(0)),
      torch.tensor(1),
      abs_state, abs_goal,
      abs_state_next,abs_goal_next
      )
    )
  decon_envs.close()

def collectData4ClassifierUsingDeconstruct(env='block_stacking', samples_per_class=250):
    if env in ['block_stacking',
             'house_building_1',
             'house_building_2',
             'house_building_3',
             'house_building_4',
             'improvise_house_building_2',
             'improvise_house_building_3',
             'improvise_house_building_discrete',
             'improvise_house_building_random',
             'ramp_block_stacking',
             'ramp_house_building_1',
             'ramp_house_building_2',
             'ramp_house_building_3',
             'ramp_house_building_4',
             'ramp_improvise_house_building_2',
             'ramp_improvise_house_building_3']:
        deconstruct_env = env + '_deconstruct'
        decon_envs = EnvWrapper(num_processes, deconstruct_env, env_config, planner_config)
    elif env in ['1l1b1r', '1l2b2r', '1l1l1r', '1l1l2r', '1l2b1r', '1l2b2b2r', '1l2b1l2b2r']:
        deconstruct_env = 'house_building_x' + '_deconstruct'
        env_config['goal_string'] = env
    else:
        raise NotImplementedError('deconstruct env not supported for env: {}'.format(env))
    print(f'==================\t Collecting {env} dataset\t===================')
    decon_envs = EnvWrapper(num_processes, deconstruct_env, env_config, planner_config)
    num_objects = decon_envs.getNumObj()
    num_classes = 2*num_objects-1
    print(f'[INFO] Num classes: {num_classes}')

    dataset = ListDataset()

    transitions = decon_envs.gatherDeconstructTransitions(samples_per_class)
    decon_envs.close()
    transitions.reverse()

    true_index = [i for i in range(len(transitions)) if transitions[i][3] is True]
    perfect_index = [true_index[i] for i in range(len(true_index)) if (true_index[i] == num_classes-2) or (true_index[i]-true_index[i-1] == num_classes-1)]
    for i in perfect_index:
        for j in range(num_classes-1, 0, -1):
            dataset.add("HAND_BITS", transitions[i-j+1][0][0])
            dataset.add("OBS", transitions[i-j+1][0][2])
            dataset.add("HAND_OBS", transitions[i-j+1][0][1])
            dataset.add("DONES", j)
            dataset.add("ABS_STATE_INDEX", j)
                
            if j == 1:
                dataset.add("HAND_BITS", transitions[i][4][0])
                dataset.add("OBS", transitions[i][4][2])
                dataset.add("HAND_OBS", transitions[i][4][1])
                dataset.add("DONES", 1)
                dataset.add("ABS_STATE_INDEX", 0)
            
    dataset = dataset.to_array_dataset({
        "HAND_BITS": np.int32, "OBS": np.float32, "HAND_OBS": np.float32,
        "DONES": bool,
        "ABS_STATE_INDEX": np.int32,
    })
    print("Number collected samples: ", dataset.size)
    dataset.save_hdf5(f"bulletarm_baselines/fc_dqn/classifiers/{env}.h5")
    print("DONE!!!")

if __name__ == '__main__':
    # argument parser
    collectData4ClassifierUsingDeconstruct(env=env, samples_per_class=samples_per_class)
