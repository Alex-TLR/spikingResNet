"""
Analyse SpikeResNet10 / SpikeResNet18 parameter counts and memory footprint
for deployment on the STM32F769 Discovery board.

STM32F769 resources
-------------------
  Internal Flash  : 2 MB
  Internal SRAM   : 512 KB  (DTCM 128 KB + SRAM1/2/3 384 KB)
  External SDRAM  : 16 MB   (IS42S32400F on Discovery board)
  QSPI NOR Flash  : up to 128 MB (board dependent)

Run:
    python stm32/model_info.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from models.spikeresnet import SpikeResNet10Model, SpikeResNet18Model

# ── STM32F769 memory map ────────────────────────────────────────────────────
HW = {
    "Internal Flash (bytes)":  2 * 1024 * 1024,
    "Internal SRAM  (bytes)":  512 * 1024,
    "External SDRAM (bytes)":  16 * 1024 * 1024,
}

BYTES_F32 = 4
BYTES_INT8 = 1


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def layer_table(model: nn.Module) -> list[dict]:
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            n = sum(p.numel() for p in module.parameters())
            rows.append({"name": name, "type": type(module).__name__, "params": n})
    return rows


def activation_peak(model_name: str, in_channels: int, num_classes: int,
                    beta: float = 0.5, threshold: float = 1.0) -> int:
    """
    Estimate peak activation memory (bytes, float32) during a single-image
    forward pass. Conservatively assumes two consecutive feature-map buffers
    must coexist in memory at the same time.
    """
    # Only Conv2D output tensors are tracked; LIF membrane shares the same shape.
    if model_name == "SpikeResNet10":
        # input 32x32
        shapes = [
            (64,  32, 32),   # after block1
            (64,  32, 32),   # after resBlock2_1/2
            (128, 16, 16),   # after resBlock4 (stride-2)
            (256, 8,  8),    # after resBlock6
            (512, 4,  4),    # after resBlock8
            (512, 1,  1),    # after amax9
        ]
    else:  # SpikeResNet18
        shapes = [
            (64,  32, 32),
            (64,  32, 32),
            (128, 16, 16),
            (256, 8,  8),
            (512, 4,  4),
            (512, 1,  1),
        ]
    peak = max(c * h * w for c, h, w in shapes)   # worst-case single buffer
    # ×2 because two adjacent buffers exist simultaneously (input + output)
    return peak * 2 * BYTES_F32


def print_report(model_name: str, model: nn.Module,
                 in_channels: int, num_classes: int) -> None:
    p = count_params(model)
    rows = layer_table(model)

    w_f32  = p["total"] * BYTES_F32
    w_int8 = p["total"] * BYTES_INT8
    act    = activation_peak(model_name, in_channels, num_classes)

    print(f"\n{'='*60}")
    print(f"  {model_name}  ({num_classes} classes, {in_channels}-ch input)")
    print(f"{'='*60}")
    print(f"  Total parameters : {p['total']:>10,}")
    print(f"  Weights float32  : {w_f32/1024/1024:>8.2f} MB  ({w_f32:,} bytes)")
    print(f"  Weights int8     : {w_int8/1024/1024:>8.2f} MB  ({w_int8:,} bytes)")
    print(f"  Peak activations : {act/1024:>8.1f} KB  (float32, batch=1)")
    print()

    print(f"  {'Layer':<35} {'Type':<16} {'Params':>10}")
    print(f"  {'-'*62}")
    for r in rows:
        print(f"  {r['name']:<35} {r['type']:<16} {r['params']:>10,}")

    print()
    print("  STM32F769 fit assessment (float32 weights):")
    for label, cap in HW.items():
        fits = "OK " if w_f32 <= cap else "TOO LARGE"
        pct  = w_f32 / cap * 100
        print(f"    {label:<28}: {fits}  ({pct:.0f}% of {cap//1024} KB)")
    print()
    print("  STM32F769 fit assessment (int8 weights):")
    for label, cap in HW.items():
        fits = "OK " if w_int8 <= cap else "TOO LARGE"
        pct  = w_int8 / cap * 100
        print(f"    {label:<28}: {fits}  ({pct:.0f}% of {cap//1024} KB)")
    print()
    print(f"  Activation memory ({act//1024} KB) vs Internal SRAM "
          f"({HW['Internal SRAM  (bytes)']//1024} KB): "
          f"{'OK' if act <= HW['Internal SRAM  (bytes)'] else 'OVERFLOW'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    beta, thresh = 0.5, 1.0

    # CIFAR-10 (3-channel, 10 classes)
    m10 = SpikeResNet10Model(numberOfChannels=3, numberOfClasses=10,
                              beta=beta, threshold=thresh)
    print_report("SpikeResNet10", m10, in_channels=3, num_classes=10)

    m18 = SpikeResNet18Model(numberOfChannels=3, numberOfClasses=10,
                              beta=beta, threshold=thresh)
    print_report("SpikeResNet18", m18, in_channels=3, num_classes=10)
