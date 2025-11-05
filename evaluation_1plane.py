#!/usr/bin/env python3
"""
Evaluation script for 1-plane Flow LSTM implementation.

NOTE: This script requires the Flow1PlaneEvaluator module to be implemented.
For now, please use evaluation_1plane_new.py which provides the modular interface.

TODO: Implement Flow1PlaneEvaluator in evaluation_modules/flow_evaluator_1plane.py
      This should follow the pattern from Flow3PlaneEvaluator but adapted for:
      - Single y-plane (yslice54)
      - 3 channels (u, v, w) instead of 12
      - 256×256 resolution instead of 128×128
      - Simplified channel organization
"""

# import warnings

# warnings.warn(
#     "evaluation_1plane.py is a placeholder. "
#     "Please use evaluation_1plane_new.py for modular evaluation, "
#     "or implement Flow1PlaneEvaluator module following the 3-plane pattern.",
#     UserWarning,
# )

if __name__ == "__main__":
    print("=" * 70)
    print("NOTICE: evaluation_1plane.py is currently a placeholder")
    print("=" * 70)
    print()
    print("To evaluate your 1-plane LSTM model, please use:")
    print("  python evaluation_1plane_new.py <checkpoint_path>")
    print()
    print("Or implement the Flow1PlaneEvaluator module at:")
    print("  evaluation_modules/flow_evaluator_1plane.py")
    print()
    print("The Flow1PlaneEvaluator should be adapted from Flow3PlaneEvaluator")
    print("with these changes:")
    print("  - num_channels: 12 → 3")
    print("  - num_planes: 3 → 1")
    print("  - y_slices: [29, 54, 75] → 54")
    print("  - field_names: ['u', 'v', 'w', 'p'] → ['u', 'v', 'w']")
    print("  - input_shape: [128, 128] → [256, 256]")
    print("  - Dataset: FlowSequence3PlaneDataset → FlowSequence1PlaneDataset")
    print("=" * 70)
