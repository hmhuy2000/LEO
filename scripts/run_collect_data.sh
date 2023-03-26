export PYTHONPATH="$PWD/"

CUDA_VISIBLE_DEVICES=0 python bulletarm_baselines/fc_dqn/scripts/fill_buffer_deconstruct.py\
 --env=block_stacking --samples_per_class=1000