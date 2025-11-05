# LSTM 1-Plane Implementation Summary

## Overview
Successfully created a complete LSTM 1-plane implementation adapted from the LSTM 3-plane model. This implementation processes single y-plane (yslice54) data with 3 velocity channels (u, v, w) at 256×256 resolution.

## Files Created

### 1. Dataset Implementation
- **File**: `src/datasets/flow_sequence_2d/flow_sequence_1plane.py` (391 lines)
- **Purpose**: Dataset loader for single-plane flow sequence data
- **Key Features**:
  - Loads 3 channels (u, v, w) from yslice54
  - 256×256 resolution (higher than 3-plane's 128×128)
  - Handles temporal discontinuity at timestep 1081
  - Per-channel normalization support
  - Data file pattern: `u-v-w_scale2-3-1_yslice54_t*.h5`

### 2. Data Configuration (4 YAML files)
- **Main config**: `configs/data/flow_sequence_1plane/flow_sequence_1plane.yaml`
  - Batch size: 10 (larger than 3-plane due to fewer channels)
  - Normalization file: `norm_stats_3ch_1plane_u-v-w_scale2-3-1_yslice54.json`

- **Train split**: `configs/data/flow_sequence_1plane/train/flow_1plane_train.yaml`
- **Validation split**: `configs/data/flow_sequence_1plane/valid/flow_1plane_valid.yaml`
- **Test split**: `configs/data/flow_sequence_1plane/test/flow_1plane_test.yaml`

### 3. Model Implementation
- **File**: `src/models/base/lstm_1plane.py` (379 lines)
- **Purpose**: ConvLSTM encoder-decoder for 1-plane flow prediction
- **Architecture**:
  - Input channels: 3 (u, v, w)
  - Input resolution: 256×256
  - Hidden dimensions: [64, 128, 256]
  - Kernel sizes: [3, 3, 3]
  - Number of layers: 3
  - Encoder-decoder structure with autoregressive prediction

### 4. Model Configuration
- **File**: `configs/model/lstm_1plane.yaml`
- **Key Parameters**:
  - input_shape: [256, 256]
  - num_channels: 3
  - sequence_length: 5
  - prediction_horizon: 1
  - hidden_dimensions: [64, 128, 256]

### 5. Training Configuration
- **File**: `configs/train_flow_lstm_1plane.yaml`
- **Settings**:
  - Task name: `flow_lstm_1plane`
  - Log project: `turbulence_lstm_1plane`
  - Tags: ["flow", "lstm", "1plane", "uvw", "3channels"]
  - Max epochs: 50
  - Validation check interval: 1.0 (every epoch)

### 6. Evaluation Scripts
- **Modular version**: `evaluation_1plane_new.py` (272 lines)
  - Uses modular Flow1PlaneEvaluator (needs implementation)
  - 9 monitoring points for single plane
  - Future steps: 100 (configurable)

- **Placeholder**: `evaluation_1plane.py`
  - Notes that full evaluator module needs implementation
  - References evaluation_1plane_new.py for usage

### 7. Training Script
- **File**: `start_train_flow_lstm_1plane_conda.sh`
- **Default settings**:
  - Max steps: 20,000
  - Validation check interval: 300 steps
  - Limited validation batches: 5
  - Includes alternative epoch-based training command

### 8. Testing Script
- **File**: `test_1plane_config.py`
- **Tests performed**:
  ✅ Configuration file validation
  ✅ Dataset instantiation (1413 training samples)
  ✅ Sample loading (input: [1,5,3,256,256], label: [1,1,3,256,256])
  ✅ Model instantiation
  ✅ Forward pass validation

## Key Differences from 3-Plane

| Aspect | 3-Plane | 1-Plane |
|--------|---------|---------|
| Channels | 12 (3 planes × 4 fields) | 3 (1 plane × 3 fields) |
| Fields | u, v, w, p | u, v, w |
| Resolution | 128×128 | 256×256 |
| Y-slices | [29, 54, 75] | 54 |
| Batch size | 6 | 10 |
| Data file | `u-v-w-p_scale4-6-1` | `u-v-w_scale2-3-1` |
| Norm stats | 12-channel | 3-channel |

## Dataset Statistics
- **Total timesteps**: 1,581
- **Valid samples**: 1,570 (6 excluded due to discontinuity)
- **Train split**: 1,413 samples (90%)
- **Val split**: 79 samples (5%)
- **Test split**: 78 samples (5%)
- **Data shape**: (3, 256, 256) per timestep

## Normalization Statistics
Per-channel normalization loaded from `norm_stats_3ch_1plane_u-v-w_scale2-3-1_yslice54.json`:
- **u channel**: mean=0.817898, std=0.112683
- **v channel**: mean=0.000000, std=0.054851
- **w channel**: mean=0.001272, std=0.074860

## Model Memory Footprint
Estimated memory per batch (batch_size=10):
- Input: ~10 × 5 × 3 × 256 × 256 × 4 bytes ≈ 37 MB
- LSTM states: ~115 MB (hidden states for encoder/decoder)
- **Total**: ~150-200 MB per batch (acceptable for training)

## How to Use

### 1. Start Training
```bash
bash start_train_flow_lstm_1plane_conda.sh
```

Or with custom parameters:
```bash
python src/train.py --config-name=train_flow_lstm_1plane \
    trainer.max_epochs=100 \
    data.batch_size_per_device=8
```

### 2. Test Configuration
```bash
python test_1plane_config.py
```

### 3. Evaluate Model
```bash
python evaluation_1plane_new.py <checkpoint_path> \
    --num-samples 3 \
    --num-future-steps 100 \
    --save-predictions
```

## Next Steps

### Required for Evaluation
The modular evaluation approach requires implementing:
- **File**: `evaluation_modules/flow_evaluator_1plane.py`
- **Class**: `Flow1PlaneEvaluator`
- **Adapt from**: `Flow3PlaneEvaluator` in `evaluation_modules/flow_evaluator_3plane.py`
- **Key changes**:
  - num_channels: 12 → 3
  - num_planes: 3 → 1
  - field_names: ['u', 'v', 'w', 'p'] → ['u', 'v', 'w']
  - Dataset: FlowSequence3PlaneDataset → FlowSequence1PlaneDataset

### Optional Enhancements
1. **Curriculum Learning**: Enable k-step rollout in config
2. **Scheduled Sampling**: Enable in config for better long-term prediction
3. **Energy Spectra Analysis**: Implement in evaluator module
4. **Multi-scale Training**: Experiment with different resolutions

## Verification Results
All configuration tests passed successfully:
```
✅ Configuration files validated
✅ Dataset instantiation successful (1413 train samples)
✅ Sample loading verified
✅ Model instantiation successful
✅ Forward pass completed (output shape: [1, 3, 256, 256])
```

## Notes
- All files follow the same patterns as 3-plane implementation
- Normalization statistics file already exists in data directory
- Training can start immediately
- Evaluation requires Flow1PlaneEvaluator implementation
- Configuration is compatible with existing infrastructure

## Git Branch
Current branch: `uvwp_3y_lstm`

**Important**: This implementation was created from scratch on this branch. The FNO 1-plane infrastructure from another branch is NOT available here.
