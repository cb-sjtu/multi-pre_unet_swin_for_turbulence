#!/usr/bin/env python
"""
Quick test script to verify FNO model configuration loads correctly with Hydra.
"""

import hydra
import torch
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="train_flow_fno_3plane", version_base="1.3")
def test_config(cfg: DictConfig):
    print("=" * 60)
    print("Testing FNO3Plane Configuration")
    print("=" * 60)

    # 1. Test model instantiation
    print("\n1. Testing model instantiation...")
    try:
        model_cfg = cfg.model
        print(f"   Model target: {model_cfg._target_}")
        print(f"   Input shape: {model_cfg.input_shape}")
        print(f"   Sequence length: {model_cfg.sequence_length}")
        print(f"   Num channels: {model_cfg.num_channels}")
        print(f"   Hidden channels: {model_cfg.hidden_channels}")
        print(f"   Num layers: {model_cfg.num_layers}")
        print(f"   Num modes: {model_cfg.num_modes}")

        # Instantiate model
        model = hydra.utils.instantiate(model_cfg)
        print("   ✓ Model instantiated successfully")
        print(f"   Model type: {type(model).__name__}")

        # Get model info
        info = model.get_model_info()
        print(f"   Total parameters: {info['total_params']:,}")
        print(f"   Trainable parameters: {info['trainable_params']:,}")

    except Exception as e:
        print(f"   ✗ Error instantiating model: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 2. Test forward pass
    print("\n2. Testing forward pass...")
    try:
        # Create dummy input
        batch_size = 2
        T, C, H, W = 5, 12, 128, 128
        x = torch.randn(batch_size, T, C, H, W)
        print(f"   Input shape: {x.shape}")

        # Forward pass
        model.eval()
        with torch.no_grad():
            y = model(x)

        print(f"   Output shape: {y.shape}")
        assert y.shape == (batch_size, C, H, W), f"Expected shape {(batch_size, C, H, W)}, got {y.shape}"
        print("   ✓ Forward pass successful")

    except Exception as e:
        print(f"   ✗ Error in forward pass: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 3. Test configuration
    print("\n3. Testing other configurations...")
    try:
        print(f"   Task name: {cfg.task_name}")
        print(f"   Log project: {cfg.log_project}")
        print(f"   Loss function: {cfg.loss_fn}")
        print(f"   Tags: {cfg.tags}")
        print("   ✓ Configuration loaded successfully")

    except Exception as e:
        print(f"   ✗ Error in configuration: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_config()
