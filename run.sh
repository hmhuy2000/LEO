# export PYTHONPATH=/home/hnguyen/long_branch/huy_new/BulletArm/:$PYTHONPATH
# export PYTHONPATH=/home/hnguyen/long_branch/house/BulletArm/:$PYTHONPATH
# export PYTHONPATH=/home/hainguyen/huy/BulletArm/:$PYTHONPATH
# export PYTHONPATH=/home/hainguyen/long_branch/BulletArm/:$PYTHONPATH
export PYTHONPATH=/home/huy/Documents/Robotics/BulletArm/:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python bulletarm_baselines/fc_dqn/scripts/main_goal.py\
 --algorithm=sdqfd --architecture=equi_asr --env=house_building_4 --fill_buffer_deconstruct\
 --planner_episode=5 --max_train_step=10000 --wandb_group=goal_5 --device_name=cuda\
 --wandb_logs=0 --use_classifier=0 --get_bad_pred=0 --seed=0 --classifier_name=perfect\
 --use_equivariant=0 --dummy_number=1

# -------------- collect data -------------- #
# python bulletarm_baselines/fc_dqn/scripts/fill_buffer_deconstruct.py

# -----------train classifier -------------- #
# CUDA_VISIBLE_DEVICES=1 python bulletarm_baselines/fc_dqn/scripts/State_abstractor.py

# python multi_run.py --file=cmd1.yaml