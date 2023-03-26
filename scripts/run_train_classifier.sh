export PYTHONPATH="$PWD/"

CUDA_VISIBLE_DEVICES=0 python bulletarm_baselines/fc_dqn/scripts/State_abstractor.py\
  --env=block_stacking --use_equivariant=True