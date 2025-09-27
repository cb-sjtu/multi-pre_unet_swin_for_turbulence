#!/usr/bin/env python3
"""
Debug script for U-net skip connections dimension matching
使用与实际训练相同的配置参数
"""

import torch

from src.models.base.swin_transformer import SwinTransformer2DWithMerging


def debug_skip_connections():
    """Debug U-net skip connections with exact training configuration"""

    # 使用与训练脚本完全相同的配置
    config = {
        "input_shape": [128, 128],
        "sequence_length": 5,
        "prediction_horizon": 1,
        "num_channels": 18,  # 6 planes × 3 fields (u,v,w)
        "patch_size": [4, 4],
        "embed_dim": 256,
        "depths": [2, 2, 4, 6, 4, 2, 2],  # encoder: [2,2,4], latent: [6], decoder: [4,2,2]
        "num_heads": 16,
        "window_size": [8, 8],
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.1,
        "drop_path_rate": 0.1,
        "patch_norm": True,
        "final_upsample": "expand_first",
        "use_patch_merging": True,
    }

    print("=== U-net Skip Connection Debug ===")
    print(f"Configuration: {config['depths']}")
    print(f"embed_dim: {config['embed_dim']}")

    # 计算encoder/decoder层数
    num_encoder_layers = len(config["depths"]) // 2  # 3
    print(f"num_encoder_layers: {num_encoder_layers}")

    # 分析encoder维度变化
    print("\n=== Encoder Dimension Analysis ===")
    encoder_dims = []
    for i in range(num_encoder_layers):
        dim = config["embed_dim"] * (2**i)
        encoder_dims.append(dim)
        print(f"Encoder Layer {i}: dim = {dim}")

    # 分析decoder维度变化和skip连接
    print("\n=== Decoder Skip Connection Analysis ===")
    num_layers_decoder = len(config["depths"]) - num_encoder_layers - 1  # 3

    for i_layer in range(num_layers_decoder):
        # 当前decoder层的输入维度
        input_dim = config["embed_dim"] * (2 ** (num_encoder_layers - 1 - i_layer))

        # PatchExpand2D输出维度 (默认减半)
        upsampled_dim = input_dim // 2

        # 计算skip连接索引
        skip_idx = num_encoder_layers - 2 - i_layer

        print(f"\nDecoder Layer {i_layer}:")
        print(f"  Input dim: {input_dim}")
        print(f"  After PatchExpand2D: {upsampled_dim}")
        print(f"  Skip connection index: {skip_idx}")

        if skip_idx >= 0 and skip_idx < len(encoder_dims):
            encoder_dim = encoder_dims[skip_idx]
            print(f"  Encoder feature dim: {encoder_dim}")
            print(f"  Concatenated dim: {upsampled_dim} + {encoder_dim} = {upsampled_dim + encoder_dim}")

            # 检查维度是否匹配
            if upsampled_dim == encoder_dim:
                print("  ✅ Dimensions match!")
            else:
                print(f"  ❌ Dimension mismatch! {upsampled_dim} != {encoder_dim}")
        else:
            print(f"  ⚠️  Invalid skip index: {skip_idx}")

    print("\n=== Creating Model Instance ===")
    try:
        model = SwinTransformer2DWithMerging(**config)
        print("✅ Model created successfully!")

        # 测试前向传播
        print("\n=== Testing Forward Pass ===")
        B, T, C, H, W = (
            1,
            config["sequence_length"],
            config["num_channels"],
            config["input_shape"][0],
            config["input_shape"][1],
        )
        test_input = torch.randn(B, T, C, H, W)
        print(f"Test input shape: {test_input.shape}")

        with torch.no_grad():
            try:
                output = model(test_input)
                print("✅ Forward pass successful!")
                print(f"Output shape: {output.shape}")
            except Exception as e:
                print(f"❌ Forward pass failed: {e}")
                print(f"Error type: {type(e).__name__}")

                # 如果是维度不匹配错误，提供详细信息
                if "Sizes of tensors must match" in str(e):
                    print("\n🔍 This appears to be a dimension mismatch error in skip connections!")
                    print("Suggested fixes:")
                    print("1. Check PatchExpand2D out_dim parameter")
                    print("2. Verify skip connection index calculation")
                    print("3. Ensure encoder features are stored at correct time")

    except Exception as e:
        print(f"❌ Model creation failed: {e}")


if __name__ == "__main__":
    debug_skip_connections()
