export PYTHONPATH="$PWD/"

CUDA_VISIBLE_DEVICES=0 python bulletarm_baselines/fc_dqn/scripts/main_goal.py\
 --algorithm=sdqfd --architecture=equi_asr --env=block_stacking --fill_buffer_deconstruct\
 --planner_episode=5 --max_train_step=10000 --wandb_group=test_5 --device_name=cuda\
 --wandb_logs=False --use_classifier=True --seed=0 --classifier_name=normal\
 --use_equivariant=True --train_phrase=True\
 --load_model_pre=None