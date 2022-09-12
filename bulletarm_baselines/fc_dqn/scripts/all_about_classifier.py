# from ast import arg
# from cProfile import label
# from cgi import test
# from turtle import title

# from matplotlib import use
# from bulletarm_baselines.fc_dqn.utils.SoftmaxClassifier import SoftmaxClassifier
# from bulletarm_baselines.fc_dqn.utils.View import View
# from bulletarm_baselines.fc_dqn.utils.ConvEncoder import ConvEncoder
# from bulletarm_baselines.fc_dqn.utils.SplitConcat import SplitConcat
# from bulletarm_baselines.fc_dqn.utils.FCEncoder import FCEncoder
# from bulletarm_baselines.fc_dqn.utils.EquiObs import EquiObs
# from bulletarm_baselines.fc_dqn.utils.EquiHandObs import EquiHandObs
# from bulletarm_baselines.fc_dqn.utils.dataset import ArrayDataset, count_objects
# from bulletarm_baselines.fc_dqn.utils.result import Result
# from sklearn.metrics import accuracy_score, f1_score, classification_report

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import copy as cp
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import seaborn as sns
# import pandas as pd
# import argparse

# def create_folder(path):
#     try:
#         os.mkdir(path)
#     except:
#         print(f'[INFO] folder {path} existed, can not create new')

# def load_dataset(goal_str, validation_fraction=0.2, test_fraction=0.1, eval=False):
#     dataset = ArrayDataset(None)
#     if eval:
#         print("=================\t Loading finetune dataset \t=================")
#         dataset.load_hdf5(f"bulletarm_baselines/fc_dqn/classifiers/{goal_str}.h5")
#         num_samples = dataset.size
#         print(f"Total number samples: {num_samples}")
#         abs_index = dataset["TRUE_ABS_STATE_INDEX"]
#         print(f"Class: {np.unique(abs_index, return_counts=True)[0]}")
#         print(f"Number samples/each class: {np.unique(abs_index, return_counts=True)[1]}")
#         return dataset
#     else:
#         print("=================\t Loading dataset \t=================")
#         dataset.load_hdf5(f"bulletarm_baselines/fc_dqn/classifiers/{goal_str}.h5")
#         dataset.shuffle()
#         num_samples = dataset.size
#         print(f"Total number samples: {num_samples}")
#         abs_index = dataset["ABS_STATE_INDEX"]
#         print(f"Class: {np.unique(abs_index, return_counts=True)[0]}")
#         print(f"Number samples/each class: {np.unique(abs_index, return_counts=True)[1]}")

#         valid_samples = int(num_samples * validation_fraction)
#         valid_dataset = dataset.split(valid_samples)
#         test_samples = int(num_samples * test_fraction)
#         test_dataset = dataset.split(test_samples)
#         dataset.size = dataset.size - valid_dataset.size - test_dataset.size
#         return dataset, valid_dataset, test_dataset


# def build_classifier(num_classes,device, use_equivariant=False):
#     """
#     Build model classifier

#     Args:
#     - num_classes
#     """

#     # encodes obs of shape Bx1x128x128 into Bx128x5x5
#     if use_equivariant:
#         print('=============================================')
#         print('----------\t Equivaraint Model \t -----------')
#         print('=============================================')
#         conv_obs = EquiObs(num_subgroups=4, filter_sizes=[3, 3, 3, 3, 3, 3], filter_counts=[32, 64, 128, 256, 512, 128])
#         conv_obs_avg_pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)

#     else:    
#         conv_obs = ConvEncoder({
#             "input_size": [128, 128, 1],
#             "filter_size": [3, 3, 3, 3, 3],
#             "filter_counts": [32, 64, 128, 256, 128],
#             "strides": [2, 2, 2, 2, 2],
#             "use_batch_norm": True,
#             "activation_last": True,
#             "flat_output": False
#         })
#         # average pool Bx128x5x5 into Bx128x1x1 and reshape that into Bx128
#         conv_obs_avg_pool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
#     conv_obs_view = View([128])
#     conv_obs_encoder = nn.Sequential(conv_obs, conv_obs_avg_pool, conv_obs_view)

#     # encodes hand obs of shape Bx1x24x24 into Bx128x1x1
#     if use_equivariant:
#         conv_hand_obs = EquiHandObs(num_subgroups=8, filter_sizes=[3, 3, 3, 3], filter_counts=[32, 64, 128, 128])
#     else:
#         conv_hand_obs = ConvEncoder({
#         "input_size": [24, 24, 1],
#         "filter_size": [3, 3, 3, 3],
#         "filter_counts": [32, 64, 128, 128],
#         "strides": [2, 2, 2, 2],
#         "use_batch_norm": True,
#         "activation_last": True,
#         "flat_output": False
#     })
#     # reshape Bx128x1x1 into Bx128
#     conv_hand_obs_view = View([128])
#     conv_hand_obs_encoder = nn.Sequential(conv_hand_obs, conv_hand_obs_view)
#     # gets [obs, hand_obs], runs that through their respective encoders
#     # and then concats [Bx128, Bx128] into Bx256
#     conv_encoder = SplitConcat([conv_obs_encoder, conv_hand_obs_encoder], 1)

#     intermediate_fc = FCEncoder({
#         "input_size": 256,
#         "neurons": [256, 256, 128],
#         "use_batch_norm": True,
#         "use_layer_norm": False,
#         "activation_last": True
#     })

#     encoder = nn.Sequential(conv_encoder, intermediate_fc, nn.Dropout(p=0.5))

#     encoder.output_size = 128

#     classifier = SoftmaxClassifier(encoder, conv_encoder, intermediate_fc, num_classes)
#     classifier.to(device)
#     return classifier


# def get_batch(epoch_step, dataset):
#     b = np.index_exp[epoch_step * batch_size: (epoch_step + 1) * batch_size]

#     obs = dataset["OBS"][b]
#     hand_obs = dataset["HAND_OBS"][b]
#     abs_state_index = dataset["ABS_STATE_INDEX"][b]
#     return torch.from_numpy(obs[:, np.newaxis, :, :]).to(device), \
#         torch.from_numpy(hand_obs[:, np.newaxis, :, :]).to(device), \
#         torch.from_numpy(abs_state_index).to(device)


# def validate(classifier, valid_dataset):
#     classifier.eval()
#     result = Result()
#     result.register("TOTAL_LOSS")
#     result.register("ACCURACY")
#     # throws away a bit of data if validation set size % batch size != 0
#     num_steps = int(len(valid_dataset["OBS"]) // batch_size)
#     for step in range(num_steps):
#         obs, hand_obs, abs_task_indices = get_batch(epoch_step=step, dataset=valid_dataset)
#         loss, acc = classifier.compute_loss_and_accuracy([obs, hand_obs], abs_task_indices)
#         result.add_pytorch("TOTAL_LOSS", loss)
#         result.add("ACCURACY", acc)
#     classifier.train()
#     return result.mean("TOTAL_LOSS"), result.mean("ACCURACY")

# def finetune_model_to_proser(finetune_epoch, finetune_learning_rate, lamda0, lamda1, lamda2):
   
#     finetune_loss = nn.CrossEntropyLoss()
#     finetune_optimizer = optim.SGD(classifier.parameters(), lr=finetune_learning_rate, momentum=0.9, weight_decay=5e-4)

#     false_count = 0
#     best_finetune_model = None
   
#     for fi_ep in range(finetune_epoch):
#         dataset.shuffle()
#         classifier.train()
        
#         train_loss = 0
#         correct = 0
#         total = 0
#         percent = []
#         for i in range(epoch_size):
#             obs, hand_obs, abs_task_indices = get_batch(epoch_step=i, dataset=dataset)
#             finetune_optimizer.zero_grad()
#             beta = torch.distributions.Beta(1, 1).sample([]).item()

#             halflength = int(len(obs)/2)
#             prehalf_obs = obs[:halflength]
#             prehalf_hand_obs = hand_obs[:halflength]
#             prehalf_label = abs_task_indices[:halflength]
#             posthalf_obs = obs[halflength:]
#             posthalf_hand_obs = hand_obs[halflength:]
#             poshalf_label = abs_task_indices[halflength:]
#             index = torch.randperm(prehalf_obs.size(0)).to(device)
#             pre2embeddings = classifier.pre2block([prehalf_obs, prehalf_hand_obs])
#             mixed_embeddings = beta*pre2embeddings + (1-beta)*pre2embeddings[index]

#             dummylogit = classifier.dummypredict([posthalf_obs, posthalf_hand_obs])
#             post_outputs = classifier.forward([posthalf_obs, posthalf_hand_obs])
#             posthalf_output = torch.cat((post_outputs, dummylogit), 1)
#             prehalf_output = torch.cat((classifier.latter2blockclf1(mixed_embeddings), classifier.latter2blockclf2(mixed_embeddings)), 1)
#             maxdummy, _ = torch.max(dummylogit.clone(), dim=1)
#             maxdummy = maxdummy.view(-1, 1)
#             dummyoutputs = torch.cat((post_outputs.clone(), maxdummy), dim=1)
#             for i in range(len(dummyoutputs)):
#                 nowlabel = poshalf_label[i]
#                 dummyoutputs[i][nowlabel] = -1e-9
#             dummytargets = torch.ones_like(poshalf_label)*num_classes
#             outputs = torch.cat((prehalf_output, posthalf_output), 0)
#             loss1 = finetune_loss(prehalf_output, (torch.ones_like(prehalf_label)*num_classes).long().to(device))
#             loss2 = finetune_loss(posthalf_output, poshalf_label.long())
#             loss3 = finetune_loss(dummyoutputs, dummytargets.long())
#             loss = lamda0 * loss1 +  lamda1 * loss2 + lamda2 * loss3
#             loss.backward()
#             finetune_optimizer.step()
#             train_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += abs_task_indices.size(0)
#             correct += predicted.eq(abs_task_indices).sum().item()
#             percent.append(correct/total)
#         percent = np.array(percent)
#         print(f"Finetune Epoch {fi_ep}: {percent.mean()}")

#         per, valid = validate_model(classifier=classifier, finetune=True)
#         print(per, valid)
#         if per > false_count and valid > 0.95:
#             false_count = per
#             best_finetune_model = cp.deepcopy(classifier.state_dict())
#             print("[INFO] Saving classifier ...")
#     if best_finetune_model is not None: 
#         classifier.load_state_dict(best_finetune_model)
#     else:
#         print("NOTHING GOOD")
        
#     return classifier    

# def validate_model(classifier, finetune=False):
#     classifier.eval()
#     correct = 0
#     unseen = 0
#     preds = []
#     # throws away a bit of data if validation set size % batch size != 0
#     for i in range(eval_dataset.size):
#         obs = torch.from_numpy(eval_dataset["OBS"][i][np.newaxis, np.newaxis, :, :]).to(device)
#         hand_obs = torch.from_numpy(eval_dataset["HAND_OBS"][i][np.newaxis, np.newaxis, :, :]).to(device)
#         # true_abs_state_index = torch.tensor(eval_dataset["TRUE_ABS_STATE_INDEX"][i]).to(device)
#         if finetune:
#             pred = classifier.proser_prediction([obs, hand_obs])
#             preds.append(pred.cpu().detach().numpy())
#         else:
#             pred = classifier.get_prediction([obs, hand_obs], logits=False, hard=True)
#             preds.append(pred.cpu().detach().numpy()[0])
#         # if pred == true_abs_state_index:
#             # correct += 1
#         # if pred == num_classes and pred == true_abs_state_index:
#             # unseen += 1
#     # print(preds)
#     print(classification_report(eval_dataset["TRUE_ABS_STATE_INDEX"], preds))
#     print('--------')
#     classifier.train()
#     return f1_score(eval_dataset["TRUE_ABS_STATE_INDEX"], preds, average='micro'), validate(classifier=classifier, valid_dataset=valid_dataset)[1]
    
# def tsne_visualize(classifier, dataset):
#     print(dataset.size)
#     # exit()
#     out = []
#     label = []
#     classifier.eval()
#     for i in range(dataset.size):
#         obs = torch.from_numpy(dataset['OBS'][i][np.newaxis, np.newaxis, :, :]).to(device)
#         inhand = torch.from_numpy(dataset['HAND_OBS'][i][np.newaxis, np.newaxis, :, :]).to(device)
#         out.append(classifier.encoder([obs, inhand]).cpu().detach().numpy().reshape(128))
#         label.append([dataset["ABS_STATE_INDEX"][i]])
#     out = np.array(out)
#     label = np.array(label).reshape(-1,)
#     print(out.shape)
#     print(label.shape)

#     tsne = TSNE(n_components=2, verbose=1, random_state=123, init='pca')
#     tsne.fit_transform(out)
#     create_folder("TSNE")
#     df = pd.DataFrame()
#     df["y"] = label
#     df['comp-1'] = out[:, 0]
#     df['comp-2'] = out[:, 1]

#     sns_plot = sns.scatterplot(x='comp-1', y='comp-2', hue=df.y.tolist(),
#                         palette=sns.color_palette("hls", num_classes), data=df).set(title=f"{goal_string}")
    
#     plt.savefig(f"TSNE/{goal_string}.png")

# def load_classifier(goal_str, num_classes, use_equivariant=False, use_proser=False, dummy_number=1,device=None):
#     classifier = build_classifier(num_classes=num_classes,device=device, use_equivariant=use_equivariant)
#     classifier.train()
#     if use_proser:
#         classifier.create_dummy(dummy_number=dummy_number)
#         classifier.to('cuda')
#         if use_equivariant:
#             classifier.load_state_dict(torch.load(f"bulletarm_baselines/fc_dqn/classifiers/finetune_equi_{goal_str}.pt"))
#         else:
#             classifier.load_state_dict(torch.load(f"bulletarm_baselines/fc_dqn/classifiers/finetune_{goal_str}.pt"))
#     else:
#         if use_equivariant:
#             classifier.load_state_dict(torch.load(f"bulletarm_baselines/fc_dqn/classifiers/equi_{goal_str}.pt"))
#         else:
#             classifier.load_state_dict(torch.load(f"bulletarm_baselines/fc_dqn/classifiers/{goal_str}.pt"))
#     classifier.to(device)
#     classifier.eval()
#     print('------\t Successfully load classifier \t-----------')
#     return classifier


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument('-gs', '--goal_str', default='block_stacking', help='The goal string task')
#     ap.add_argument('-bs', '--batch_size', default=32, help='Number of samples in a batch')
#     ap.add_argument('-nts', '--num_training_steps', default=10000, help='Number of training step')
#     ap.add_argument('-dv', '--device', default='cuda:0', help='Having gpu or not')
#     ap.add_argument('-lr', '--learning_rate', default=1e-3, help='Learning rate')
#     ap.add_argument('-wd', '--weight_decay', default=1e-5, help='Weight decay')
#     ap.add_argument('-ufm', '--use_equivariant', default=False, help='Using equivariant or not')
#     ap.add_argument('-up', '--use_proser', default=False, help='Using Proser (open-set recognition) or not')
#     ap.add_argument('-dn', '--dummy_number', default=5, help='Number of dummy classifiers')
#     ap.add_argument('-fep', '--finetune_epoch', default=30, help='Number of finetune epoch')
#     ap.add_argument('-ld0', '--lamda0', default=0.01, help='Weight for data placeholder loss')
#     ap.add_argument('-ld1', '--lamda1', default=1, help='Weight for classifier placeholder loss (mapping the nearest to ground truth label)')
#     ap.add_argument('-ld2', '--lamda2', default=1, help='Weight for classifier placeholder loss (mapping the second nearest to the dummpy classifier )')
#     ap.add_argument('-grs', '--grid_search', default=False, help='grid_search')


#     args = vars(ap.parse_args())
#     goal_string = args['goal_str']
#     batch_size = args['batch_size']
#     device = torch.device(args['device'])
#     proser = args['use_proser']

#     if args['goal_str'] == 'block_stacking':
#         num_objects = 4
#     elif args['goal_str'] == 'house_building_1':
#         num_objects = 4
#     elif args['goal_str'] == 'house_building_2':
#         num_objects = 3
#     elif args['goal_str'] == 'house_building_3':
#         num_objects = 4
#     elif args['goal_str'] == 'house_building_4':
#         num_objects = 6
#     else:
#         num_objects = count_objects(goal_string)
#     num_classes = 2 * num_objects - 1
#     print("=================================")
#     print("Training classifier for task: {:s} goal, {:d} objects".format(goal_string, num_objects))
#     print("=================================")
     
#     # Load dataset
#     dataset, valid_dataset, test_dataset = load_dataset(goal_str=goal_string)
#     epoch_size = dataset["OBS"].shape[0] // batch_size

#     # eval_dataset = load_dataset(goal_str="eval_house_building_2", eval=True)
#     # eval_dataset = load_dataset(goal_str='eval_house_building_2_dqn_equi_classifier', eval=True)
#     # eval_dataset = load_dataset(goal_str='training_cls_house_building_2_dqn_classifier', eval=True)
#     # eval_dataset = load_dataset(goal_str='training_cls_house_building_2_dqn_equi_classifier', eval=True)

#     # classifier = load_classifier(goal_str=goal_string, num_classes=num_classes, use_proser=proser, dummy_number=5, use_equivariant=args['use_equivariant'])
#     # eval_online, _ = validate_model(classifier=classifier, finetune=True)
#     # print(f"Eval Acc: ", eval_online )
#     # exit()
#     # tsne_visualize(classifier=classifier, dataset=dataset)
#     # exit()
#     # Build model
#     classifier = build_classifier(num_classes=num_classes, use_equivariant=args['use_equivariant'])
#     classifier.train()
#     if proser:
#         if args['use_equivariant']:
#             classifier.load_state_dict(torch.load(f"bulletarm_baselines/fc_dqn/classifiers/equi_{goal_string}.pt"))
#         else:
#             classifier.load_state_dict(torch.load(f"bulletarm_baselines/fc_dqn/classifiers/{goal_string}.pt"))
#         classifier.create_dummy(dummy_number=args['dummy_number'])
#     classifier.to(device)

#     if not proser:
#           # Init optimizer
#         params = classifier.parameters()
#         print("num parameter tensors: {:d}".format(len(list(classifier.parameters()))))
#         opt = optim.Adam(params, lr=args['learning_rate'], weight_decay=args['weight_decay'])
#         best_val_loss, best_classifier = None, None

#         result = Result()
#         result.register("TOTAL_LOSS")
#         result.register("ACCURACY")
#         result.register("TOTAL_VALID_LOSS")
#         result.register("VALID_ACCURACY")

#         for training_step in range(args['num_training_steps']):
#             epoch_step = training_step % epoch_size
#             if epoch_step == 0:
#                 dataset.shuffle()
#             if training_step % 300 == 0:
#                 valid_loss, valid_acc = validate(classifier=classifier, valid_dataset=valid_dataset)
#                 if best_val_loss is None or best_val_loss > valid_loss:
#                     best_val_loss = valid_loss
#                     best_classifier = cp.deepcopy(classifier.state_dict())
#                 result.add("TOTAL_VALID_LOSS", valid_loss)
#                 result.add("VALID_ACCURACY", valid_acc)
#                 print("validation complete")
#             if training_step % 100 == 0:
#                 print("step {:d}".format(training_step))
#             obs, hand_obs, abs_task_indices = get_batch(epoch_step=epoch_step, dataset=dataset)
#             opt.zero_grad()
#             loss, acc = classifier.compute_loss_and_accuracy([obs, hand_obs], abs_task_indices)
#             loss.backward()
#             opt.step()
#             result.add_pytorch("TOTAL_LOSS", loss)
#             result.add("ACCURACY", acc)

#         if best_classifier is not None:
#             classifier.load_state_dict(best_classifier)
#         else:
#             print("Best model not saved.")
#         losses = np.stack(result["TOTAL_LOSS"], axis=0)
#         valid_losses = np.stack(result["TOTAL_VALID_LOSS"], axis=0)
#         acc = np.stack(result["ACCURACY"], axis=0)
#         valid_acc = np.stack(result["VALID_ACCURACY"], axis=0)

#         # Plot Loss and Acc curve
#         create_folder('Loss_and_Acc')
#         plt.figure(figsize=(8, 6))
#         x = np.arange(0, valid_losses.shape[0])
#         x *= 300

#         plt.subplot(3, 1, 1)
#         plt.plot(losses, linestyle='-', color='blue', label="Training loss")
#         plt.plot(x, valid_losses, linestyle='--', color='red', marker='*', label='Valid loss')
#         plt.legend()

#         plt.subplot(3, 1, 2)
#         plt.plot(losses, linestyle='-', color='blue', label="Training loss (log)")
#         plt.plot(x, valid_losses, linestyle='--', color='red', marker='*', label='Valid loss (log)')
#         plt.yscale('log')
#         plt.legend()

#         plt.subplot(3, 1, 3)
#         plt.plot(acc, linestyle='-', color='blue', label='Training acc')
#         plt.plot(x, valid_acc, linestyle='--', color='red', marker='*', label='Valid acc')
#         plt.legend()

#         if args['use_equivariant']:
#             name = 'equi_' + goal_string
#         else:
#             name = goal_string
#         plt.savefig(f'Loss_and_Acc/{name}.png')

#         classifier.eval()
#         final_valid_loss = validate(classifier=classifier, valid_dataset=valid_dataset)
#         print(f" Valid Loss: {final_valid_loss[0]} and Valid Accuracy: {final_valid_loss[1]}")
#         test_loss = validate(classifier=classifier, valid_dataset=test_dataset)
#         print(f"Test Loss: {test_loss[0]} and Test Accuracy: {test_loss[1]}")

#         # eval_online, _ = validate_model(classifier=classifier, finetune=False)
#         # print(f"Eval Acc: ", eval_online )

        
#         if not args['use_equivariant']:
#             torch.save(classifier.state_dict(), f"bulletarm_baselines/fc_dqn/classifiers/{goal_string}.pt")
#         else:
#             torch.save(classifier.state_dict(), f"bulletarm_baselines/fc_dqn/classifiers/equi_{goal_string}.pt")
        
#     elif args['grid_search']:
#         for dc in [5, 10, 15]:    
#             for lamda0 in [0.001, 0.01, 0.1]:
#                 for lamda2 in [0.1, 0.5, 0.8, 1, 1.2, 1.5, 2.0]:
#                     lamda1 = 1.0
#                     print('=*50')
#                     print(f"Dummy class: {dc}, lamda0: {lamda0}, lamda1: {lamda1}, lamda2: {lamda2}")
#                     if args['use_equivariant']:
#                         classifier.load_state_dict(torch.load(f"bulletarm_baselines/fc_dqn/classifiers/equi_{goal_string}.pt"))
#                     else:
#                         classifier.load_state_dict(torch.load(f"bulletarm_baselines/fc_dqn/classifiers/{goal_string}.pt"))
#                     classifier.train()
#                     classifier.create_dummy(dummy_number=dc).to(device)
#                     classifier = finetune_model_to_proser(finetune_epoch=args['finetune_epoch'], finetune_learning_rate=args['learning_rate'], lamda0=lamda0, lamda1=lamda1, lamda2=lamda2)
#     else:
#         classifier = finetune_model_to_proser(finetune_epoch=args['finetune_epoch'], finetune_learning_rate=args['learning_rate'], lamda0=args['lamda0'], lamda1=args['lamda1'], lamda2=args['lamda2'])
#         if not args['use_equivariant']:
#             torch.save(classifier.state_dict(), f"bulletarm_baselines/fc_dqn/classifiers/finetune_{goal_string}.pt")
#         else:
#             torch.save(classifier.state_dict(), f"bulletarm_baselines/fc_dqn/classifiers/finetune_equi_{goal_string}.pt")
