from re import L
import matplotlib
matplotlib.use('Agg')
import time
import os
import sys
import time
import copy
import math
import collections
from tqdm import tqdm
from datetime import datetime
import threading

import torch
from torch import nn
import wandb

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
sys.path.append('./')
sys.path.append('..')
from bulletarm_baselines.fc_dqn.scripts.create_agent import createAgent
from bulletarm_baselines.fc_dqn.storage.buffer import QLearningBufferExpert, QLearningBuffer
from bulletarm_baselines.logger.logger import Logger
from bulletarm_baselines.logger.baseline_logger import BaselineLogger
from bulletarm_baselines.fc_dqn.utils.schedules import LinearSchedule
from bulletarm_baselines.fc_dqn.utils.env_wrapper import EnvWrapper

from bulletarm_baselines.fc_dqn.utils.parameters import *
from bulletarm_baselines.fc_dqn.utils.torch_utils import augmentBuffer, augmentBufferD4
from bulletarm_baselines.fc_dqn.scripts.fill_buffer_deconstruct import train_fillDeconstructUsingRunner

from bulletarm_baselines.fc_dqn.scripts.State_abstractor import State_abstractor
from bulletarm_baselines.fc_dqn.utils.dataset import ListDataset, count_objects


ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert abs_state abs_goal abs_state_next abs_goal_next')

def create_folder(path):
    try:
        os.mkdir(path)
    except:
        print(f'[INFO] folder {path} existed, can not create new')

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def getCurrentObs(in_hand, obs):
    obss = []
    for i, o in enumerate(obs):
        obss.append((o.squeeze(), in_hand[i].squeeze()))
    return obss

def update_abs_goals(abs_states):
    with torch.no_grad():
        zeros_goals = torch.zeros_like(abs_states)
        return torch.max(abs_states - 1, zeros_goals)

def remove_outlier(abs_states,num_classes):
    with torch.no_grad():
        max_abs = torch.full(abs_states.shape,num_classes - 1,dtype=abs_states.dtype).to(device)
        return torch.min(abs_states, max_abs)

def train_step(agent, replay_buffer, logger):
    batch = replay_buffer.sample(batch_size)
    loss, td_error = agent.update(batch)
    logger.logTrainingStep(loss)
    if logger.num_steps % target_update_freq == 0:
        agent.updateTarget()

def saveModelAndInfo(logger, agent):
    logger.writeLog()
    logger.exportData()
    agent.saveModel(os.path.join(logger.models_dir, 'snapshot'))

def get_cls(state_abstractor, obs, inhand):
    obs = obs.clone().detach().type(torch.cuda.FloatTensor).to(device)
    inhand = inhand.clone().detach().type(torch.cuda.FloatTensor).to(device)
    res = state_abstractor.classifier([obs,inhand]).clone().detach().type(torch.cuda.FloatTensor).to(device)
    return torch.argmax(res,dim=1)

def evaluate(envs, agent,num_eval_episodes,logger=None, wandb_logs=False,state_abstractor=None,num_steps=0,debug = False,render=False):
    num_objects = envs.getNumObj()
    num_classes = 2 * num_objects - 1 
    states, in_hands, obs = envs.reset()
    evaled = 0
    total_return = 0
    temp_reward = [[] for _ in range(num_eval_processes)]
    if not no_bar:
        eval_bar = tqdm(total=num_eval_episodes)
    
    if debug:
        dataset = ListDataset()
        create_folder(f'outlier/eval_debug_outlier_{env}_{wandb_group}')
        create_folder(f'miss/eval_debug_miss_{env}_{wandb_group}')

    cnt = 0
    while evaled < num_eval_episodes:
        true_abs_states = torch.tensor(envs.get_true_abs_states()).to(device)
        pred_abs_states = get_cls(state_abstractor, obs, in_hands)
        if (use_classifier):
            abs_states = pred_abs_states 
        else:
            abs_states = true_abs_states 
        abs_states = remove_outlier(abs_states,num_classes)
        abs_goals = update_abs_goals(abs_states)
        if (render):
            print(abs_states,abs_goals)
            time.sleep(1)

        if (debug):
            for i in range(abs_states.shape[0]):
                if (true_abs_states[i] != pred_abs_states[i]):
                    dataset.add("HAND_BITS", states[i].cpu().detach().numpy().astype(np.int32))
                    dataset.add("OBS", obs[i].reshape(128,128).cpu().detach().numpy().astype(np.float32))
                    dataset.add("HAND_OBS", in_hands[i].reshape(24,24).cpu().detach().numpy().astype(np.float32))
                    dataset.add("TRUE_ABS_STATE_INDEX", true_abs_states[i].cpu().detach().numpy().astype(np.int32))
                    dataset.add("PRED_ABS_STATE_INDEX", pred_abs_states[i].cpu().detach().numpy().astype(np.int32))
                    plt.figure(figsize=(15,4))
                    plt.subplot(1,2,1)
                    plt.imshow(obs[i].reshape(128,128), cmap='gray')
                    plt.colorbar()
                    plt.subplot(1,2,2)
                    plt.imshow(in_hands[i].reshape(24,24), cmap='gray')
                    plt.colorbar()
                    plt.suptitle(f"True: {true_abs_states[i]}, Pred: {pred_abs_states[i]}")
                    if (true_abs_states[i].cpu().detach().numpy().astype(np.int32) == num_classes):
                        plt.savefig(f'outlier/eval_debug_outlier_{env}_{wandb_group}/image_{cnt}.png')
                    else:
                        plt.savefig(f'miss/eval_debug_miss_{env}_{wandb_group}/image_{cnt}.png')
                    plt.close()
                    cnt += 1

        q_value_maps, actions_star_idx, actions_star = agent.getEGreedyActions(states, in_hands, obs,abs_states,abs_goals, 0)
        actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
        states_, in_hands_, obs_, rewards, dones = envs.step(actions_star, auto_reset=True)


        rewards = rewards.numpy()
        dones = dones.numpy()
        states = copy.copy(states_)
        in_hands = copy.copy(in_hands_)
        obs = copy.copy(obs_)
        for i, r in enumerate(rewards.reshape(-1)):
            temp_reward[i].append(r)
        evaled += int(np.sum(dones))
        for i, d in enumerate(dones.astype(bool)):
            if d:
                R = 0
                for r in reversed(temp_reward[i]):
                    R = r + gamma * R
                if (logger):
                    logger.logEvalEpisode(temp_reward[i], discounted_return=R)
                total_return += temp_reward[i][-1]
                if(render):
                    print('return is',temp_reward[i][-1])

                temp_reward[i] = []
        if not no_bar:
            eval_bar.update(evaled - eval_bar.n)

    if (debug):
        dataset = dataset.to_array_dataset({
            "HAND_BITS": np.int32, "OBS": np.float32, "HAND_OBS": np.float32,
            "TRUE_ABS_STATE_INDEX": np.int32,"PRED_ABS_STATE_INDEX": np.int32,
        })
        dataset.metadata = {
            "NUM_EXP": dataset.size, "TIMESTAMP": str(datetime.today())
        }
        print('get',dataset.size,'data samples')
        dataset.save_hdf5(f"bulletarm_baselines/fc_dqn/data/eval_data_{env}_{wandb_group}.h5")

    print(f'evaluate results: {total_return/num_eval_episodes}')
    Wandb_logging(f'mean evaluate return',total_return/num_eval_episodes,num_steps,wandb_logs)
    if (logger):
        logger.logEvalInterval()
        logger.writeLog()
    if not no_bar:
        eval_bar.close()
    
def Wandb_logging(key, value, step_idx,wandb_logs):
    if (wandb_logs):
        try:
            wandb.log({key:value},step = step_idx)
        except:
            print(f'[INFO] {key}: {value}')
    else:
        print(f'[INFO] {key}: {value}')

def train():
    print(f'trainning for {max_train_step} step on {env}')
    if (wandb_logs):
        print('---------------------using Wandb---------------------')
        wandb.init(project=env, settings=wandb.Settings(_disable_stats=True), \
        group=wandb_group, name=wandb_seed, entity='hmhuy')
    else:
        print('----------------------no Wandb-----------------------')

    eval_thread = None
    start_time = time.time()
    if seed is not None:
        set_seed(seed)
    # setup env
    if env in ['1l1l1r', '1l1l2r', '1l2b2r', '1l2b1r', '1l2b2b2r', '1l2b1l2b2r']:
        env_config['goal_string'] = env
        envs = EnvWrapper(num_processes, 'house_building_x', env_config, planner_config)
        eval_envs = EnvWrapper(num_eval_processes, 'house_building_x', env_config, planner_config)
    else:    
        envs = EnvWrapper(num_processes, env, env_config, planner_config)
        eval_envs = EnvWrapper(num_eval_processes, env, env_config, planner_config)

    num_objects = envs.getNumObj()
    num_classes = 2 * num_objects - 1 
    print(f'num class = {num_classes}')
    state_abstractor = State_abstractor(goal_str=env, use_equivariant=use_equivariant, device=device)
    state_abstractor.load_classifier()
    agent = createAgent(num_classes)
    eval_agent = createAgent(num_classes,test=True)
    # load classifier
    if (use_classifier):
        print('---- use abstract state from classifier  ----')
    else:
        print('---- use true abstract state from environment    -----')
    

    if load_model_pre is not None:
        print(f'load load agent from {load_model_pre}')
        agent.loadModel(load_model_pre)
    else:
        print('training from scratch')
    agent.train()
    eval_agent.train()

    # logging
    base_dir = os.path.join(log_pre, f'{env}_{wandb_group}')
    if note:
        base_dir += '_'
        base_dir += note
    if not log_sub:
        timestamp = time.time()
        timestamp = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d.%H:%M:%S')
        log_dir = os.path.join(base_dir, timestamp)
    else:
        log_dir = os.path.join(base_dir, log_sub)

    hyper_parameters['model_shape'] = agent.getModelStr()
    logger = BaselineLogger(log_dir, checkpoint_interval=save_freq, num_eval_eps=num_eval_episodes, hyperparameters=hyper_parameters, eval_freq=eval_freq)
    logger.saveParameters(hyper_parameters)

    if buffer_type == 'expert':
        replay_buffer = QLearningBufferExpert(buffer_size)
    else:
        raise NotImplementedError('buffer type in ["expert"]')
    exploration = LinearSchedule(schedule_timesteps=explore, initial_p=init_eps, final_p=final_eps)
    print(f'explore scheduler for {explore} steps, from {init_eps} to {final_eps}')

    states, in_hands, obs = envs.reset()

    if load_sub:
        logger.loadCheckPoint(os.path.join(base_dir, load_sub, 'checkpoint'), agent.loadFromState, replay_buffer.loadFromState)
    #------------------------------------- expert transition ----------------------------------------#    
    if planner_episode > 0 and not load_sub:
        if fill_buffer_deconstruct:
            train_fillDeconstructUsingRunner(agent, replay_buffer)
    #------------------------------------- pretrainning with expert ----------------------------------------#    
    #-------------------------------------- start trainning ----------------------------------------------#
    if not no_bar:
        pbar = tqdm(total=max_train_step)
        pbar.set_description('Episodes:0; Reward:0.0; Explore:0.0; Loss:0.0; Time:0.0')
    timer_start = time.time()
    total_goal = np.zeros((num_classes))
    success_goal = np.zeros((num_classes))
    old_total_goal = np.zeros((num_classes))
    old_success_goal = np.zeros((num_classes))
    train_return = []
    if (get_bad_pred):
        dataset = ListDataset()
        # create_folder(f'outlier/debug_outlier_{env}_{wandb_group}')
        create_folder('miss/')
        create_folder(f'miss/debug_miss_{env}_{wandb_group}')
        cnt = 0
    while logger.num_training_steps < max_train_step + 1:
        if (logger.num_training_steps%eval_freq == 0 and logger.num_training_steps > 0):
            for idx in range(num_classes-1):
                Wandb_logging(f'number of goal {idx}',total_goal[idx] - old_total_goal[idx],logger.num_training_steps,wandb_logs)
                Wandb_logging(f'number of achieved goal {idx}',success_goal[idx] - old_success_goal[idx],logger.num_training_steps,wandb_logs)
                Wandb_logging(f'success rate of goal {idx}',(success_goal[idx]-old_success_goal[idx])/(total_goal[idx]-old_total_goal[idx]),logger.num_training_steps,wandb_logs)
            Wandb_logging(f'mean training return',np.mean(train_return),logger.num_training_steps,wandb_logs)
            old_total_goal = copy.deepcopy(total_goal)
            old_success_goal = copy.deepcopy(success_goal)
            train_return = []
        if fixed_eps:
            eps = final_eps
        else:
            eps = exploration.value(logger.num_eps)
        is_expert = 0
        true_abs_states = torch.tensor(envs.get_true_abs_states()).to(device)
        pred_abs_states = get_cls(state_abstractor, obs, in_hands)
        if (use_classifier):
            abs_states = pred_abs_states 
        else:
            abs_states = true_abs_states 

        #############################################
        if (get_bad_pred and logger.num_training_steps<=5000):
            for i in range(abs_states.shape[0]):
                if (true_abs_states[i].cpu().detach().numpy().astype(np.int32) != pred_abs_states[i].cpu().detach().numpy().astype(np.int32)) and (true_abs_states[i].cpu().detach().numpy().astype(np.int32) != num_classes):
                    assert true_abs_states[i].cpu().detach().numpy() != pred_abs_states[i].cpu().detach().numpy()
                    dataset.add("HAND_BITS", states[i].cpu().detach().numpy().astype(np.int32))
                    dataset.add("OBS", obs[i].reshape(128,128).cpu().detach().numpy().astype(np.float32))
                    dataset.add("HAND_OBS", in_hands[i].reshape(24,24).cpu().detach().numpy().astype(np.float32))
                    dataset.add("TRUE_ABS_STATE_INDEX", true_abs_states[i].cpu().detach().numpy().astype(np.int32))
                    dataset.add("PRED_ABS_STATE_INDEX", pred_abs_states[i].cpu().detach().numpy().astype(np.int32))
                    plt.figure(figsize=(15,4))
                    plt.subplot(1,2,1)
                    plt.imshow(obs[i].reshape(128,128), cmap='gray')
                    plt.colorbar()
                    plt.subplot(1,2,2)
                    plt.imshow(in_hands[i].reshape(24,24), cmap='gray')
                    plt.colorbar()
                    plt.suptitle(f"True: {true_abs_states[i]}, Pred: {pred_abs_states[i]}")
                    # if (true_abs_states[i].cpu().detach().numpy().astype(np.int32) == num_classes):
                        # plt.savefig(f'outlier/debug_outlier_{env}_{wandb_group}/image_{cnt}.png')
                    # else:
                    plt.savefig(f'miss/debug_miss_{env}_{wandb_group}/image_{cnt}.png')
                    plt.close()
                    cnt += 1
            if (logger.num_training_steps == 5000):
                dataset = dataset.to_array_dataset({
                    "HAND_BITS": np.int32, "OBS": np.float32, "HAND_OBS": np.float32,
                    "TRUE_ABS_STATE_INDEX": np.int32,"PRED_ABS_STATE_INDEX": np.int32,
                })
                dataset.metadata = {
                "NUM_EXP": dataset.size, "TIMESTAMP": str(datetime.today())
                }
                print('get',dataset.size,'data samples')
                dataset.save_hdf5(f"bulletarm_baselines/fc_dqn/classifiers/train_data_{env}_{wandb_group}.h5")
                exit()
        #############################################

        abs_states = remove_outlier(abs_states,num_classes)
        abs_goals = update_abs_goals(abs_states)
        for i in range(num_processes):
            total_goal[abs_goals[i]] += 1

        q_value_maps, actions_star_idx, actions_star = agent.getEGreedyActions(
            states, in_hands, obs,abs_states,abs_goals, eps
            )

        buffer_obs = getCurrentObs(in_hands, obs)
        actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
        envs.stepAsync(actions_star, auto_reset=False)
        if len(replay_buffer) >= training_offset:
            for training_iter in range(training_iters):
                train_step(agent, replay_buffer, logger)

        states_, in_hands_, obs_, rewards, dones = envs.stepWait()
        clone_rewards = copy.deepcopy(rewards)
        true_abs_states_next = torch.tensor(envs.get_true_abs_states()).to(device) 
        pred_abs_states_next = get_cls(state_abstractor, obs_, in_hands_)
        if (use_classifier):
            abs_states_next = pred_abs_states_next 
        else:
            abs_states_next = true_abs_states_next 
        abs_states_next = remove_outlier(abs_states_next,num_classes)
        abs_goals_next =  update_abs_goals(abs_states_next)
        goals_achieved = (abs_states_next == abs_goals)
        rewards = goals_achieved.unsqueeze(1).float() - 1.0 
        goals_achieved = goals_achieved.cpu()
        for i in range(num_processes):
            if goals_achieved[i].cpu().item() is False:
                dones[i] = 1.0
            else:
                if (abs_goals[i] == 0):
                    rewards[i] = clone_rewards[i]# - 1.0
                    goals_achieved[i] = (rewards[i] > 0)
                    if (goals_achieved[i].cpu().item() is True):
                        success_goal[abs_goals[i]] += 1
                    continue

                if (dones[i]>0):
                    goals_achieved[i] = not goals_achieved[i]
                    rewards[i] = torch.tensor(-1).float()
                else:
                    success_goal[abs_goals[i]] += 1

                    
        rewards = rewards.cpu()
        done_idxes = torch.nonzero(dones).squeeze(1)
        if done_idxes.shape[0] != 0:
            reset_states_, reset_in_hands_, reset_obs_ = envs.reset_envs(done_idxes)
            for j, idx in enumerate(done_idxes):
                train_return.append(clone_rewards[idx])
                states_[idx] = reset_states_[j]
                in_hands_[idx] = reset_in_hands_[j]
                obs_[idx] = reset_obs_[j]

        buffer_obs_ = getCurrentObs(in_hands_, obs_)

        for i in range(num_processes):
            replay_buffer.add(
                ExpertTransition(states[i], buffer_obs[i], actions_star_idx[i], rewards[i], 
                                states_[i], buffer_obs_[i], 
                                #------------------#
                                # dones[i], 
                                goals_achieved[i], 
                                #------------------#
                                torch.tensor(100), torch.tensor(is_expert), 
                                abs_states[i], abs_goals[i],
                                abs_states_next[i],abs_goals_next[i])
            )

        logger.logStep(clone_rewards.cpu().numpy(), dones.cpu().numpy())

        states = copy.copy(states_)
        obs = copy.copy(obs_)
        in_hands = copy.copy(in_hands_)

        if (time.time() - start_time)/3600 > time_limit:
            break

        if not no_bar:
            timer_final = time.time()
            description = 'Env Step:{}; Episode: {}; Return:{:.03f}; Eval Return:{:.03f}; Explore:{:.02f}; Loss:{:.03f}; train_steps:{}'.format(
              logger.num_steps, logger.num_eps, logger.getAvg(logger.training_eps_rewards, 100),
              np.mean(logger.eval_eps_rewards[-2]) if len(logger.eval_eps_rewards) > 1 and len(logger.eval_eps_rewards[-2]) > 0 else 0, eps, float(logger.getCurrentLoss()),
              logger.num_training_steps)
            pbar.set_description(description)
            timer_start = timer_final
            pbar.update(logger.num_training_steps - pbar.n)

        if logger.num_training_steps > 0 and eval_freq > 0 and logger.num_training_steps % eval_freq == 0:
            if eval_thread is not None:
                eval_thread.join()
            eval_agent.copyNetworksFrom(agent)
            eval_thread = threading.Thread(target=evaluate, args=(eval_envs, eval_agent,num_eval_episodes,logger, wandb_logs, state_abstractor,logger.num_training_steps,False))
            eval_thread.start()

        if logger.num_steps % (num_processes * save_freq) == 0:
            saveModelAndInfo(logger, agent)

    if eval_thread is not None:
        eval_thread.join()

    saveModelAndInfo(logger, agent)
    logger.saveCheckPoint(agent.getSaveState(), replay_buffer.getSaveState())
    envs.close()
    eval_envs.close()

def train_phrase_func():
    print('---------------------    trainning phrase    -------------------------')
    train()

def valid_phrase_func():
    print('---------------------    evaluate phrase     -------------------------')
    if (load_model_pre == 'None'):
        print('no weight path')
        return
    render = False
    env_config['render'] = render
    eval_envs = EnvWrapper(1, env, env_config, planner_config)
    num_objects = eval_envs.getNumObj()
    num_classes = 2 * num_objects - 1 
    state_abstractor = State_abstractor(goal_str=env, use_equivariant=use_equivariant, device=device)
    state_abstractor.load_classifier()

    eval_agent = createAgent(num_classes,test=True)
    eval_agent.train()
    if load_model_pre:
        eval_agent.loadModel(load_model_pre)
    eval_agent.eval()
    evaluate(envs=eval_envs,agent=eval_agent,num_eval_episodes=100, state_abstractor=state_abstractor, debug=False,render=render)
    eval_envs.close()

if __name__ == '__main__':

    if (train_phrase):
        train_phrase_func()
    else:
        valid_phrase_func()

