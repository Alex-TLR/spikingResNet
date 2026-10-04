"""
Validate that the C-exported weights produce outputs numerically equivalent
to the PyTorch model on a small set of real test images.

Steps performed
---------------
1. Load PyTorch model and fold BN.
2. Load the exported weights.bin and reconstruct the folded model in Python
   (re-loading via numpy as a cross-check).
3. Run both on the same CIFAR-10 test batch and compare membrane voltages.
4. Report max absolute error and top-1 accuracy match.

Usage
-----
    python stm32/validate_export.py \\
        --weights     weights/spike/exp1/resnet10_weights_CIFAR10_T_1_E_100_L_mse_count_loss_A_True_S_42.pth \\
        --weights-bin stm32/c_src/weights/weights.bin \\
        --dataset     CIFAR10
"""

import sys, os, struct, argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
from models.spikeresnet import SpikeResNet10Model
from stm32.bn_fold import fold_model
from stm32.export_weights import LAYER_ORDER, _get_module, NUM_CLASSES_MAP

# Number of float32 params per layer (determined by model architecture)
# We reconstruct this from the model itself, not hard-coded.


def load_folded_from_bin(model_template: torch.nn.Module,
                          bin_path: str) -> torch.nn.Module:
    """
    Re-populate *model_template* (already BN-folded) from the binary blob.
    The order must match LAYER_ORDER in export_weights.py.
    """
    with open(bin_path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.float32)

    offset = 0
    for sym, path in LAYER_ORDER:
        mod = _get_module(model_template, path)
        for tensor_name in ("weight", "bias"):
            tensor = getattr(mod, tensor_name).data
            n = tensor.numel()
            arr = torch.from_numpy(raw[offset:offset + n].copy())
            tensor.copy_(arr.reshape(tensor.shape))
            offset += n

    assert offset == len(raw), (
        f"Binary size mismatch: consumed {offset}, file has {len(raw)} floats"
    )
    return model_template


def compare_outputs(model_pt: torch.nn.Module,
                    model_bin: torch.nn.Module,
                    images: torch.Tensor,
                    num_steps: int = 1) -> dict:
    model_pt.eval()
    model_bin.eval()
    with torch.no_grad():
        _, _, mem_pt,  _ = model_pt(images,  num_steps)
        _, _, mem_bin, _ = model_bin(images, num_steps)

    # mem shape: [T, B, classes] — average over T
    mem_pt  = mem_pt.mean(0)
    mem_bin = mem_bin.mean(0)

    max_err = (mem_pt - mem_bin).abs().max().item()
    rel_err = max_err / (mem_pt.abs().max().item() + 1e-8)

    pred_pt  = mem_pt.argmax(dim=1)
    pred_bin = mem_bin.argmax(dim=1)
    top1_match = (pred_pt == pred_bin).float().mean().item()

    return {
        "max_abs_error": max_err,
        "rel_error":     rel_err,
        "top1_match":    top1_match,
    }


def load_cifar10_batch(n_images: int = 64):
    """Load n_images from the local CIFAR-10 test set (torchvision)."""
    try:
        import torchvision
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        ds = torchvision.datasets.CIFAR10(
            root="data/cifar10", train=False, download=False, transform=transform)
        loader = torch.utils.data.DataLoader(ds, batch_size=n_images, shuffle=False)
        images, labels = next(iter(loader))
        return images, labels
    except Exception as e:
        print(f"  WARNING: Could not load CIFAR-10 ({e}), using random input.")
        return torch.randn(n_images, 3, 32, 32), torch.zeros(n_images, dtype=torch.long)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",     required=True, help=".pth checkpoint")
    p.add_argument("--weights-bin", required=True, help="Exported weights.bin")
    p.add_argument("--dataset",     default="CIFAR10",
                   choices=list(NUM_CLASSES_MAP.keys()))
    p.add_argument("--channels",    type=int, default=3)
    p.add_argument("--beta",        type=float, default=0.5)
    p.add_argument("--threshold",   type=float, default=1.0)
    p.add_argument("--n-images",    type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    nC   = NUM_CLASSES_MAP[args.dataset]

    # ── Load PyTorch model ───────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.weights}")
    model_pt = SpikeResNet10Model(
        numberOfChannels=args.channels,
        numberOfClasses=nC,
        beta=args.beta,
        threshold=args.threshold,
    )
    ckpt  = torch.load(args.weights, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model_pt.load_state_dict(state)
    model_pt = fold_model(model_pt)

    # ── Reconstruct from binary ──────────────────────────────────────────────
    print(f"Loading exported binary: {args.weights_bin}")
    model_bin = SpikeResNet10Model(
        numberOfChannels=args.channels,
        numberOfClasses=nC,
        beta=args.beta,
        threshold=args.threshold,
    )
    model_bin = fold_model(model_bin)   # prepare the fold structure
    model_bin = load_folded_from_bin(model_bin, args.weights_bin)

    # ── Compare ──────────────────────────────────────────────────────────────
    print(f"Loading {args.n_images} test images …")
    images, labels = load_cifar10_batch(args.n_images)

    print("Running forward passes …")
    r = compare_outputs(model_pt, model_bin, images)

    print(f"\n  Max absolute error : {r['max_abs_error']:.4e}")
    print(f"  Relative error     : {r['rel_error']:.4e}")
    print(f"  Top-1 prediction   : {'MATCH' if r['top1_match'] == 1.0 else f\"MISMATCH ({r['top1_match']*100:.1f}% agree)\"}")

    threshold = 1e-5
    if r["max_abs_error"] < threshold and r["top1_match"] == 1.0:
        print(f"\n  EXPORT VALIDATED (max err < {threshold})\n")
    else:
        print(f"\n  WARNING: error exceeds {threshold} — check LAYER_ORDER\n")


if __name__ == "__main__":
    main()
