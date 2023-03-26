from bulletarm_baselines.fc_dqn.utils.SoftmaxClassifier import SoftmaxClassifier
from bulletarm_baselines.fc_dqn.utils.View import View
from bulletarm_baselines.fc_dqn.utils.ConvEncoder import ConvEncoder, CNNOBSEncoder, CNNHandObsEncoder
from bulletarm_baselines.fc_dqn.utils.SplitConcat import SplitConcat
from bulletarm_baselines.fc_dqn.utils.FCEncoder import FCEncoder
from bulletarm_baselines.fc_dqn.utils.EquiObs import EquiObs
from bulletarm_baselines.fc_dqn.utils.EquiHandObs import EquiHandObs
from bulletarm_baselines.fc_dqn.utils.dataset import ArrayDataset, count_objects
from bulletarm_baselines.fc_dqn.utils.result import Result
from bulletarm_baselines.fc_dqn.utils.parameters import *

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report

def create_folder(path):
    try:
        os.mkdir(path)
    except:
        print(f'[INFO] folder {path} existed, can not create new')

def load_dataset(goal_str, validation_fraction=0.75):
    dataset = ArrayDataset(None)
    print(f"=================\t Loading training dataset {goal_str}\t=================")
    dataset.load_hdf5(f"bulletarm_baselines/fc_dqn/classifiers/{goal_str}.h5")
    dataset.shuffle()
    num_samples = dataset.size
    print(f"Total number samples: {num_samples}")
    abs_index = dataset["ABS_STATE_INDEX"]
    print(f"Class: {np.unique(abs_index, return_counts=True)[0]}")
    print(f"Number samples/each class: {np.unique(abs_index, return_counts=True)[1]}")
    valid_samples = int(num_samples * validation_fraction)
    valid_dataset = dataset.split(valid_samples)
    dataset.size = dataset.size - valid_dataset.size
    return dataset, valid_dataset

class State_abstractor():
    def __init__(self, goal_str=None, use_equivariant=None, device=None):
        self.goal_str = goal_str
        self.use_equivariant = use_equivariant
        self.device = device
        self.batch_size = 32

        if self.goal_str == 'block_stacking':
            num_objects = 4
        elif self.goal_str == 'house_building_1':
            num_objects = 4
        elif self.goal_str == 'house_building_2':
            num_objects = 3
        elif self.goal_str == 'house_building_3':
            num_objects = 4
        elif self.goal_str == 'house_building_4':
            num_objects = 6
        else:
            num_objects = count_objects(self.goal_str)
        self.num_classes = 2 * num_objects - 1

        if self.use_equivariant:
            self.name = 'equi_' + self.goal_str
        else:
            self.name = self.goal_str

        self.build_state_abstractor()

    def build_state_abstractor(self):

        if self.use_equivariant:
            print('='*50)
            print('----------\t Equivaraint State Abstractor \t -----------')
            print('='*50)
            conv_obs = EquiObs(num_subgroups=4)
        else:    
            conv_obs = CNNOBSEncoder()
        conv_hand_obs = CNNHandObsEncoder()

        conv_obs_view = View([128])
        conv_obs_encoder = nn.Sequential(conv_obs, conv_obs_view)

        conv_hand_obs_view = View([128])
        conv_hand_obs_encoder = nn.Sequential(conv_hand_obs, conv_hand_obs_view)

        conv_encoder = SplitConcat([conv_obs_encoder, conv_hand_obs_encoder], 1)

        intermediate_fc = FCEncoder({
            "input_size": 256,
            "neurons": [256, 128],
            "use_batch_norm": True,
            "use_layer_norm": False,
            "activation_last": True
        })

        encoder = nn.Sequential(conv_encoder, intermediate_fc, nn.Dropout(p=0.1))
        encoder.output_size = 128

        self.classifier = SoftmaxClassifier(encoder, self.num_classes, conv_encoder)
        self.classifier.to(self.device)
        return self.classifier

    def train_state_abstractor(self, num_training_steps=10000, learning_rate=1e-3, weight_decay=1e-5, visualize=True):
        self.classifier.train()
        # Load dataset
        self.dataset, self.valid_dataset = load_dataset(goal_str=self.goal_str)
        epoch_size = len(self.dataset['OBS']) // self.batch_size
        print(f'Number of trainable parameters: {sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)}')
        opt = optim.Adam(self.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
        best_val_loss, best_classifier = None, None
        best_step = None

        result = Result()
        result.register("TOTAL_LOSS")
        result.register("ACCURACY")
        result.register("TOTAL_VALID_LOSS")
        result.register("VALID_ACCURACY")
        
        for training_step in range(num_training_steps):
            epoch_step = training_step % epoch_size
            if epoch_step == 0:
                self.dataset.shuffle()
            
            if training_step % 500 == 0:
                valid_loss, valid_acc = self.validate(dataset=self.valid_dataset)
                if best_val_loss is None or best_val_loss > valid_loss:
                    best_val_loss = valid_loss
                    best_step = training_step
                    best_classifier = cp.deepcopy(self.classifier.state_dict())
                    print(f'step {training_step}: best = {best_val_loss} at {best_step}')
                result.add("TOTAL_VALID_LOSS", valid_loss)
                result.add("VALID_ACCURACY", valid_acc)
                print("validation complete")
            
            if training_step % 100 == 0:
                print("step {:d}".format(training_step))
            
            opt.zero_grad()
            obs, hand_obs, abs_task_indices = self.get_batch(epoch_step=epoch_step, dataset=self.dataset)
            loss, acc = self.classifier.compute_loss_and_accuracy([obs, hand_obs], abs_task_indices)
            loss.backward()
            opt.step()
            result.add_pytorch("TOTAL_LOSS", loss)
            result.add("ACCURACY", acc)

        if best_classifier is not None:
            self.classifier.load_state_dict(best_classifier)
        else:
            print("Best model not saved.")

        self.classifier.eval()
        final_valid_loss = self.validate(dataset=self.valid_dataset)
        print(f"Best Valid Loss: {final_valid_loss[0]} and Best Valid Accuracy: {final_valid_loss[1]}")
        print('Saving classifier...')
        torch.save(self.classifier.state_dict(), f"bulletarm_baselines/fc_dqn/classifiers/{self.name}.pt")   
        print('Classifier saved.')
        self.classifier.eval()
        if visualize:
            self.plot_result(result)

    def validate(self, dataset):
        self.classifier.eval()
        result = Result()
        result.register("TOTAL_LOSS")
        result.register("ACCURACY")
        # throws away a bit of data if validation set size % batch size != 0
        num_steps = int(len(dataset["OBS"]) // self.batch_size)
        for step in range(num_steps):
            obs, hand_obs, abs_task_indices = self.get_batch(epoch_step=step, dataset=dataset)
            loss, acc = self.classifier.compute_loss_and_accuracy([obs, hand_obs], abs_task_indices)
            result.add_pytorch("TOTAL_LOSS", loss)
            result.add("ACCURACY", acc)
        self.classifier.train()
        return result.mean("TOTAL_LOSS"), result.mean("ACCURACY")

    def get_batch(self, epoch_step, dataset):
        b = np.index_exp[epoch_step * self.batch_size: (epoch_step + 1) * self.batch_size]

        obs = dataset["OBS"][b]
        hand_obs = dataset["HAND_OBS"][b]
        abs_state_index = dataset["ABS_STATE_INDEX"][b]

        return torch.from_numpy(obs[:, np.newaxis, :, :]).to(self.device), \
            torch.from_numpy(hand_obs[:, np.newaxis, :, :]).to(self.device), \
            torch.from_numpy(abs_state_index).to(self.device)

    def plot_result(self, result):
        losses = np.stack(result["TOTAL_LOSS"], axis=0)
        valid_losses = np.stack(result["TOTAL_VALID_LOSS"], axis=0)
        acc = np.stack(result["ACCURACY"], axis=0)
        valid_acc = np.stack(result["VALID_ACCURACY"], axis=0)

        # Plot Loss and Acc curve
        create_folder('Loss_and_Acc')
        plt.figure(figsize=(10, 10))
        x = np.arange(0, valid_losses.shape[0])
        x *= 500

        plt.subplot(3, 1, 1)
        plt.plot(losses, linestyle='-', color='blue', label="Training loss")
        plt.plot(x, valid_losses, linestyle='--', color='red', marker='*', label='Valid loss')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(losses, linestyle='-', color='blue', label="Training loss (log)")
        plt.plot(x, valid_losses, linestyle='--', color='red', marker='*', label='Valid loss (log)')
        plt.yscale('log')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(acc, linestyle='-', color='blue', label='Training acc')
        plt.plot(x, valid_acc, linestyle='--', color='red', marker='*', label='Valid acc')
        plt.legend()

        plt.savefig(f'Loss_and_Acc/{self.name}.png')
        plt.close()

    def load_classifier(self):
        self.classifier.train()
        self.classifier.load_state_dict(torch.load(f"bulletarm_baselines/fc_dqn/classifiers/{self.name}.pt"))
        self.classifier.eval()
        print(f'------\t Successfully load classifier {self.name}\t-----------')

if __name__ == '__main__':
    model = State_abstractor(goal_str=env, use_equivariant=use_equivariant, device=torch.device('cuda'))
    model.train_state_abstractor(num_training_steps=12000, visualize=False)