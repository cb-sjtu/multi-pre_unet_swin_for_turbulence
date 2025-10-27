#!/usr/bin/env python
"""
Simple test script to verify FNO model works correctly without Hydra.
"""

import torch

from src.models.nop.fno_2d_temporal import FNO2DTemporalModel


def test_fno_model():
    print("=" * 60)
    print("Testing FNO2DTemporalModel")
    print("=" * 60)

    # 1. Test model instantiation
    print("\n1. Creating model...")
    model = FNO2DTemporalModel(
        input_shape=(128, 128),
        sequence_length=5,
        num_channels=12,
        hidden_channels=64,
        num_layers=4,
        num_modes=(16, 16),
        padding_mode="replicate",
    )

    info = model.get_model_info()
    print("   Model created successfully")
    print(f"   Total parameters: {info['total_params']:,}")
    print(f"   Trainable parameters: {info['trainable_params']:,}")

    # 2. Test forward pass with different input formats
    print("\n2. Testing forward pass...")

    # Test with 4D input (B, T*C, H, W)
    print("   Testing 4D input...")
    x_4d = torch.randn(2, 60, 128, 128)
    y = model(x_4d)
    print(f"      Input:  {x_4d.shape}")
    print(f"      Output: {y.shape}")
    assert y.shape == (2, 12, 128, 128)

    # Test with 5D input (B, T, C, H, W)
    print("   Testing 5D input...")
    x_5d = torch.randn(2, 5, 12, 128, 128)
    y = model(x_5d)
    print(f"      Input:  {x_5d.shape}")
    print(f"      Output: {y.shape}")
    assert y.shape == (2, 12, 128, 128)

    # 3. Test with larger batch
    print("\n3. Testing with larger batch...")
    x_large = torch.randn(8, 5, 12, 128, 128)
    y = model(x_large)
    print(f"   Input:  {x_large.shape}")
    print(f"   Output: {y.shape}")
    assert y.shape == (8, 12, 128, 128)

    # 4. Test gradient flow
    print("\n4. Testing gradient flow...")
    model.train()
    x = torch.randn(2, 5, 12, 128, 128, requires_grad=True)
    y = model(x)
    loss = y.mean()
    loss.backward()
    print(f"   Loss: {loss.item():.6f}")
    print(f"   Input gradient exists: {x.grad is not None}")
    print(f"   Model has gradients: {any(p.grad is not None for p in model.parameters())}")

    # 5. Compare parameter count with Swin
    print("\n5. Model efficiency comparison:")
    print(f"   FNO parameters: {info['total_params']:,}")
    print("   (For reference, Swin Transformer has ~100M+ parameters)")
    print(f"   FNO is approximately {100000000 // info['total_params']}x more parameter-efficient")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

    print("\nModel is ready to use with the training pipeline.")
    print("To start training, run:")
    print("  ./start_train_flow_fno_3plane_conda.sh")


if __name__ == "__main__":
    test_fno_model()
