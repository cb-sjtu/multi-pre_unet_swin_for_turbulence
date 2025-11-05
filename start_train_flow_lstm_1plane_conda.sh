#!/bin/bash

# Training script for 1-plane Flow LSTM model
# Single y-plane (yslice54) with 3 channels (u, v, w) at 256×256 resolution

# Activate conda environment if needed
# conda activate your_env_name

# Run training with Hydra configuration
python src/train.py --config-name=train_flow_lstm_1plane \
    trainer.max_steps=20000 \
    trainer.val_check_interval=300 \
    trainer.limit_val_batches=5 \
    print_config=true

# Alternative: Train for specific number of epochs
# python src/train.py --config-name=train_flow_lstm_1plane \
#     trainer.max_epochs=50 \
#     trainer.val_check_interval=1.0 \
#     trainer.check_val_every_n_epoch=5 \
#     print_config=true

# Notes:
# - Model: LSTM 1-plane (3 channels: u,v,w)
# - Resolution: 256×256
# - Data: /home/sh/CB/icon-thewell-dev/data/preprocessed_flow/u-v-w_scale2-3-1_yslice54_*.h5
# - Normalization: norm_stats_3ch_1plane_u-v-w_scale2-3-1_yslice54.json
# - Architecture: Same as 3-plane but with 3 input/output channels instead of 12
