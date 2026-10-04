"""
Fold BatchNorm2d parameters into the preceding Conv2d layer.

For each Conv → BN pair the folded weights are:

    w_fold  = w  * (gamma / sqrt(var + eps))          [shape: Cout, Cin, kH, kW]
    b_fold  = (0 - mean) * gamma / sqrt(var + eps) + beta   [shape: Cout]

The BN layer is then effectively an identity (gamma=1, beta=0, mean=0, var=1).

Usage:
    from stm32.bn_fold import fold_bn_into_conv, fold_model
    folded_model = fold_model(model)
    w, b = fold_bn_into_conv(conv, bn)
"""

import torch
import torch.nn as nn
import copy


def fold_bn_into_conv(conv: nn.Conv2d,
                      bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (weight, bias) after fusing BN statistics into the conv kernel.
    The conv may have bias=False; the output bias always incorporates BN offset.
    """
    gamma  = bn.weight.data                         # [Cout]
    beta   = bn.bias.data                           # [Cout]
    mean   = bn.running_mean                        # [Cout]
    var    = bn.running_var                         # [Cout]
    eps    = bn.eps

    scale = gamma / torch.sqrt(var + eps)           # [Cout]

    # Reshape for broadcasting over [Cout, Cin, kH, kW]
    s = scale.view(-1, 1, 1, 1)
    w_fold = conv.weight.data * s

    # b_conv may not exist (bias=False)
    b_conv = conv.bias.data if conv.bias is not None else torch.zeros_like(mean)
    b_fold = (b_conv - mean) * scale + beta

    return w_fold, b_fold


def _is_conv_bn(modules: list) -> bool:
    return (len(modules) == 2
            and isinstance(modules[0], nn.Conv2d)
            and isinstance(modules[1], nn.BatchNorm2d))


def fold_sequential(seq: nn.Sequential) -> nn.Sequential:
    """
    If *seq* is a [Conv2d, BatchNorm2d] Sequential (as built by convBlock),
    return a new Sequential with BN folded into the conv and BN replaced by
    Identity.
    """
    children = list(seq.children())
    if not _is_conv_bn(children):
        return seq  # nothing to fold

    conv, bn = children
    w_fold, b_fold = fold_bn_into_conv(conv, bn)

    new_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True,                 # bias now carries BN offset
    )
    new_conv.weight = nn.Parameter(w_fold)
    new_conv.bias   = nn.Parameter(b_fold)

    return nn.Sequential(new_conv, nn.Identity())


def fold_model(model: nn.Module) -> nn.Module:
    """
    Deep-copy *model* and fold all Conv→BN pairs inside nn.Sequential blocks
    (i.e. every convBlock produced by SpikeResNet architectures).
    Returns the modified copy; the original is unchanged.
    """
    m = copy.deepcopy(model)
    m.eval()   # freeze BN running stats

    for name, module in m.named_children():
        if isinstance(module, nn.Sequential):
            setattr(m, name, fold_sequential(module))
        else:
            # Recurse one level (convBlocks are direct children of the model)
            for subname, submodule in module.named_children():
                if isinstance(submodule, nn.Sequential):
                    setattr(module, subname, fold_sequential(submodule))

    return m


def verify_fold(original: nn.Module, folded: nn.Module,
                input_shape: tuple = (1, 3, 32, 32),
                num_steps: int = 1,
                device: str = "cpu",
                rtol: float = 1e-4) -> bool:
    """
    Run both models on a random input and compare output logits.
    Returns True if the maximum relative difference is within *rtol*.
    """
    original.eval()
    folded.eval()
    x = torch.randn(input_shape, device=device)
    with torch.no_grad():
        spk_orig, feat_orig, mem_orig, _ = original(x, num_steps)
        spk_fold, feat_fold, mem_fold, _ = folded(x, num_steps)
    diff = (mem_orig - mem_fold).abs().max().item()
    ref  = mem_orig.abs().max().item() + 1e-8
    ok   = (diff / ref) < rtol
    print(f"  BN-fold verify: max |Δmem| = {diff:.2e}  "
          f"(ref = {ref:.2e})  → {'PASS' if ok else 'FAIL'}")
    return ok
