"""
Layer-by-layer architecture comparison:
    SpikeResNet10Model  vs  spike_resnet10(...)
    SpikeResNet18Model  vs  spike_resnet18(...)

No forward pass, no GPU required — pure structural inspection.
Run with any Python environment that has torch + snntorch installed.
"""

import sys
import torch.nn as nn

sys.path.insert(0, "/home/alex/spikingResNet")

from models.spikeresnet import (
    SpikeResNet10Model,
    SpikeResNet18Model,
    spike_resnet10,
    spike_resnet18,
)

# ── helpers ──────────────────────────────────────────────────────────────────

def leaf_modules(model):
    """Return list of (name, module) for every leaf (no children) in order."""
    return [(n, m) for n, m in model.named_modules() if not list(m.children())]


def layer_summary(module):
    """One-line description of a leaf module."""
    cls = type(module).__name__
    params = {n: tuple(p.shape) for n, p in module.named_parameters(recurse=False)}
    if params:
        param_str = "  " + "  ".join(f"{n}:{s}" for n, s in params.items())
    else:
        # snntorch / activation layers: show relevant scalar attributes
        attrs = []
        for attr in ("beta", "threshold", "reset_mechanism", "kernel_size",
                     "stride", "padding", "num_features", "in_features",
                     "out_features", "in_channels", "out_channels"):
            v = getattr(module, attr, None)
            if v is not None:
                if hasattr(v, "item"):
                    v = round(v.item(), 4)
                attrs.append(f"{attr}={v}")
        param_str = "  " + "  ".join(attrs) if attrs else ""
    return f"{cls}{param_str}"


def compare(name_a, model_a, name_b, model_b):
    leaves_a = leaf_modules(model_a)
    leaves_b = leaf_modules(model_b)

    n   = max(len(leaves_a), len(leaves_b))
    col = 52

    print(f"\n{'='*112}")
    print(f"  {name_a:<{col}}  {name_b}")
    print(f"{'='*112}")
    print(f"  {'#':<4} {'layer name':<28} {'type / params':<{col-32}}  {'layer name':<28} {'type / params'}")
    print(f"  {'-'*108}")

    mismatches = []

    for i in range(n):
        if i < len(leaves_a):
            na, ma = leaves_a[i]
            sa = layer_summary(ma)
        else:
            na, ma, sa = "—", None, "MISSING"

        if i < len(leaves_b):
            nb, mb = leaves_b[i]
            sb = layer_summary(mb)
        else:
            nb, mb, sb = "—", None, "MISSING"

        match = (
            ma is not None and mb is not None
            and type(ma) == type(mb)
            and {pn: tuple(p.shape) for pn, p in ma.named_parameters(recurse=False)}
             == {pn: tuple(p.shape) for pn, p in mb.named_parameters(recurse=False)}
        )

        flag  = "  " if match else "!!"
        left  = f"{na:<28} {sa}"[:col]
        right = f"{nb:<28} {sb}"
        print(f"{flag} {i+1:<4} {left:<{col}}  {right}")

        if not match:
            mismatches.append(i + 1)

    print(f"  {'-'*108}")

    ca = sum(p.numel() for p in model_a.parameters())
    cb = sum(p.numel() for p in model_b.parameters())
    param_ok = ca == cb

    if not mismatches:
        print(f"  Layers : ALL {n} match ✓")
    else:
        print(f"  Layers : {n - len(mismatches)}/{n} match  —  mismatches at positions {mismatches}")

    print(f"  Params : {name_a}={ca:,}   {name_b}={cb:,}   "
          f"{'MATCH ✓' if param_ok else 'MISMATCH !!'}")
    print(f"{'='*112}\n")

    return not mismatches and param_ok


# ── build models ─────────────────────────────────────────────────────────────

COMMON = dict(numberOfChannels=3, numberOfClasses=10,
              beta=0.95, threshold=0.25, numberOfSteps=1, expansion=1)


if __name__ == "__main__":
    results = []

    results.append(compare(
        "SpikeResNet10Model",  SpikeResNet10Model(**COMMON).eval(),
        "spike_resnet10(...)", spike_resnet10(**COMMON).eval(),
    ))

    results.append(compare(
        "SpikeResNet18Model",  SpikeResNet18Model(**COMMON).eval(),
        "spike_resnet18(...)", spike_resnet18(**COMMON).eval(),
    ))

    print("SUMMARY:", "ALL PASS" if all(results) else "SOME FAILED")
    sys.exit(0 if all(results) else 1)
