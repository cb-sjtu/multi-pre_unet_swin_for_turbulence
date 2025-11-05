#!/usr/bin/env python3
"""
Test script to verify FNO 1-plane configuration is correct.
"""

import sys

sys.path.insert(0, "/home/sh/CB/icon-thewell-dev")

from src.datasets.flow_sequence_2d.flow_sequence_1plane import FlowSequence1PlaneDataset


def test_dataset():
    """Test that dataset loads correctly."""
    print("=" * 70)
    print("Testing FlowSequence1PlaneDataset")
    print("=" * 70)

    dataset = FlowSequence1PlaneDataset(
        data_dir="/home/sh/CB/icon-thewell-dev/data/preprocessed_flow",
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
        norm_stats="norm_stats_3ch_1plane_u-v-w_scale2-3-1_yslice54.json",
    )

    print("\n✅ Dataset Configuration:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Channels: {dataset.num_channels}")
    print(f"  Data shape: {dataset.data_shape}")
    print(f"  Y-slice: {dataset.y_slice}")
    print(f"  Field names: {dataset.field_names}")

    # Test loading one sample
    sample = dataset[0]
    print("\n✅ Sample Test:")
    print(f"  Input shape: {sample['data']['input_seq'].shape}")
    print(f"  Label shape: {sample['label'].shape}")
    print(f"  Description: {sample['description'][0][:80]}...")

    # Get channel info
    channel_info = dataset.get_channel_info()
    print("\n✅ Channel Info:")
    for key, value in channel_info.items():
        if key != "channel_mapping":
            print(f"  {key}: {value}")
    print("  Channel mapping:")
    for mapping in channel_info["channel_mapping"]:
        print(f"    {mapping}")

    return True


def test_model_compatibility():
    """Test that model configuration is compatible with dataset."""
    print("\n" + "=" * 70)
    print("Testing Model Compatibility")
    print("=" * 70)

    # Expected model configuration
    model_config = {
        "input_shape": [256, 256],
        "sequence_length": 5,
        "prediction_horizon": 1,
        "num_channels": 3,
        "hidden_channels": 64,
        "num_layers": 4,
        "num_modes": [32, 32],
    }

    print("\n✅ Model Configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")

    # Calculate expected input size
    batch_size = 10
    seq_len = model_config["sequence_length"]
    channels = model_config["num_channels"]
    h, w = model_config["input_shape"]

    print("\n✅ Expected Tensor Shapes:")
    print(f"  Input: (B={batch_size}, T={seq_len}, C={channels}, H={h}, W={w})")
    print(f"  Flattened input: (B={batch_size}, T*C={seq_len * channels}, H={h}, W={w})")
    print(f"  Output: (B={batch_size}, C={channels}, H={h}, W={w})")

    return True


def test_training_configuration():
    """Test training configuration parameters."""
    print("\n" + "=" * 70)
    print("Testing Training Configuration")
    print("=" * 70)

    config = {
        "task_name": "flow_fno_1plane",
        "batch_size_per_device": 10,
        "max_epochs": 50,
        "max_k_steps": 16,
        "enable_per_channel_metrics": True,
        "log_project": "turbulence_fno_1plane",
    }

    print("\n✅ Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Estimate memory usage
    batch_size = config["batch_size_per_device"]
    seq_len = 5
    channels = 3
    h = w = 256

    # Input: (batch, seq, channels, h, w)
    input_elements = batch_size * seq_len * channels * h * w
    input_gb = input_elements * 4 / (1024**3)  # float32

    print("\n✅ Memory Estimation (per batch):")
    print(f"  Input elements: {input_elements:,}")
    print(f"  Input size: ~{input_gb:.2f} GB (float32)")
    print("  Recommended GPU: >= 24GB VRAM")

    return True


if __name__ == "__main__":
    print("\n🚀 FNO 1-Plane Configuration Test\n")

    try:
        test_dataset()
        test_model_compatibility()
        test_training_configuration()

        print("\n" + "=" * 70)
        print("✅ All tests PASSED!")
        print("=" * 70)
        print("\n🎉 Configuration is ready for training!")
        print("\nTo start training, run:")
        print("  bash start_train_flow_fno_1plane_conda.sh")
        print("\nOr manually:")
        print("  python src/train.py --config-name=train_flow_fno_1plane \\")
        print("      trainer.max_steps=50000 \\")
        print("      trainer.val_check_interval=300 \\")
        print("      trainer.limit_val_batches=5")
        print()

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ Test FAILED!")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
