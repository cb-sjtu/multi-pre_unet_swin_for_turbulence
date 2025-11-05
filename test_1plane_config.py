#!/usr/bin/env python3
"""Quick test script to verify 1-plane LSTM configuration."""

import sys

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def test_1plane_config():
    """Test loading 1-plane configuration."""
    print("=" * 70)
    print("Testing 1-plane LSTM Configuration")
    print("=" * 70)

    try:
        # Test 1: Load model config (simplified - no hydra)
        print("\n1. Testing model config files...")
        import os

        config_files = [
            "configs/train_flow_lstm_1plane.yaml",
            "configs/model/lstm_1plane.yaml",
            "configs/data/flow_sequence_1plane/flow_sequence_1plane.yaml",
            "configs/data/flow_sequence_1plane/train/flow_1plane_train.yaml",
        ]

        for config_file in config_files:
            if os.path.exists(config_file):
                print(f"✓ Found: {config_file}")
            else:
                print(f"✗ Missing: {config_file}")
                return False

        # Load and validate YAML files
        from omegaconf import OmegaConf

        model_cfg = OmegaConf.load("configs/model/lstm_1plane.yaml")
        print("\n✓ Model config loaded")
        print(f"  - Model type: {model_cfg._target_}")
        print(f"  - Input shape: {model_cfg.input_shape}")
        print(f"  - Num channels: {model_cfg.num_channels}")
        print(f"  - Hidden dims: {model_cfg.hidden_dimensions}")

        train_cfg = OmegaConf.load("configs/train_flow_lstm_1plane.yaml")
        print("\n✓ Training config loaded")
        print(f"  - Task name: {train_cfg.task_name}")
        print(f"  - Log project: {train_cfg.log_project}")

    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    try:
        # Test 2: Instantiate dataset
        print("\n2. Testing dataset instantiation...")
        from src.datasets.flow_sequence_2d.flow_sequence_1plane import FlowSequence1PlaneDataset

        data_dir = "/home/sh/CB/icon-thewell-dev/data/preprocessed_flow"
        norm_stats_file = "norm_stats_3ch_1plane_u-v-w_scale2-3-1_yslice54.json"

        test_dataset = FlowSequence1PlaneDataset(
            data_dir=data_dir,
            input_length=5,
            max_k_steps=1,
            field_names=["u", "v", "w"],
            file_pattern="*u-v-w_scale2-3-1_yslice54_*.h5",
            resolution_scale=(2, 3, 1),
            y_slice=54,
            train_ratio=0.9,
            valid_ratio=0.05,
            test_ratio=0.05,
            split="train",
            enable_normalization=True,
            norm_stats=norm_stats_file,
        )

        print(f"✓ Dataset created successfully with {len(test_dataset)} samples")
        print(f"  - Data shape: {test_dataset.data_shape}")
        print(f"  - Num channels: {test_dataset.num_channels}")
        print(f"  - Y-slice: {test_dataset.y_slice}")
        print(f"  - Field names: {test_dataset.field_names}")

    except Exception as e:
        print(f"✗ Dataset instantiation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    try:
        # Test 3: Load a sample
        print("\n3. Testing sample loading...")
        sample = test_dataset[0]
        input_seq = sample["data"]["input_seq"]
        label = sample["label"]

        print("✓ Sample loaded successfully")
        print(f"  - Input shape: {input_seq.shape}")
        print(f"  - Label shape: {label.shape}")
        print(f"  - Description: {sample['description'][0]}")

    except Exception as e:
        print(f"✗ Sample loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    try:
        # Test 4: Instantiate model
        print("\n4. Testing model instantiation...")
        import torch

        from src.models.base.lstm_1plane import LSTM1Plane

        model = LSTM1Plane(
            input_shape=[256, 256],
            sequence_length=5,
            prediction_horizon=1,
            num_channels=3,
            hidden_dims=[64, 128, 256],
            kernel_sizes=[3, 3, 3],
            num_layers=3,
        )

        print("✓ Model created successfully")
        print(f"  - Model type: {model.__class__.__name__}")
        print(f"  - Input shape: {model.input_shape}")
        print(f"  - Num channels: {model.num_channels}")
        print(f"  - Sequence length: {model.sequence_length}")

        # Test forward pass
        print("\n5. Testing model forward pass...")
        with torch.no_grad():
            output = model(input_seq)
            print("✓ Forward pass successful")
            print(f"  - Output shape: {output.shape}")

    except Exception as e:
        print(f"✗ Model instantiation/forward failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("✅ All tests passed! 1-plane LSTM configuration is valid.")
    print("=" * 70)
    print("\nYou can now start training with:")
    print("  bash start_train_flow_lstm_1plane_conda.sh")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = test_1plane_config()
    sys.exit(0 if success else 1)
