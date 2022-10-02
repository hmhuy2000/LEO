# export PYTHONPATH=/path/to/LEO/:$PYTHONPATH

export PYTHONPATH=/home/huy/Documents/Robotics/LEO/:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python bulletarm_baselines/fc_dqn/scripts/fill_buffer_deconstruct.py\
 --env=house_building_1