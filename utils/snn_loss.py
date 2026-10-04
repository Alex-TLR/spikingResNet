"""Loss adapters for temporal and population-coded SNN outputs."""

import torch
import torch.nn.functional as F


CE_SOURCES = ("logits", "membranes", "spikes")
CE_MODES = (
    "temporal_mean",
    "per_timestep",
    "final_timestep",
    "spike_rate",
    "spike_count",
)
POPULATION_REDUCTIONS = ("mean", "sum")

_SOURCE_MODES = {
    "logits": {"temporal_mean", "per_timestep", "final_timestep"},
    "membranes": {"temporal_mean", "per_timestep", "final_timestep"},
    "spikes": {
        "temporal_mean",
        "per_timestep",
        "final_timestep",
        "spike_rate",
        "spike_count",
    },
}


def validate_ce_options(source, mode, population_reduction):
    """Validate a cross-entropy source, temporal mode, and population reduction."""
    if source not in CE_SOURCES:
        raise ValueError(f"Unsupported CE source: {source}. Expected one of {CE_SOURCES}.")
    if mode not in CE_MODES:
        raise ValueError(f"Unsupported CE mode: {mode}. Expected one of {CE_MODES}.")
    if population_reduction not in POPULATION_REDUCTIONS:
        raise ValueError(
            "Unsupported population reduction: "
            f"{population_reduction}. Expected one of {POPULATION_REDUCTIONS}."
        )
    if mode not in _SOURCE_MODES[source]:
        raise ValueError(f"CE mode '{mode}' is not compatible with source '{source}'.")


def resolve_ce_options(source, mode, population_reduction, fit):
    """Resolve backward-compatible CE defaults from the legacy fit mode."""
    resolved_source = source or ("membranes" if fit == "membrane" else "spikes")
    resolved_mode = mode or (
        "temporal_mean" if resolved_source in ("logits", "membranes") else "spike_rate"
    )
    resolved_reduction = population_reduction or "sum"
    validate_ce_options(resolved_source, resolved_mode, resolved_reduction)
    return resolved_source, resolved_mode, resolved_reduction


def aggregate_population(trace, num_classes, reduction="mean"):
    """Reduce contiguous output-neuron populations to class traces."""
    if trace.ndim != 3:
        raise ValueError(
            "Expected a time-major trace with shape [T, B, outputs], "
            f"received {tuple(trace.shape)}."
        )
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer.")
    if reduction not in POPULATION_REDUCTIONS:
        raise ValueError(
            f"Unsupported population reduction: {reduction}. "
            f"Expected one of {POPULATION_REDUCTIONS}."
        )

    num_outputs = trace.size(-1)
    if num_outputs % num_classes:
        raise ValueError(
            f"Output width {num_outputs} must be divisible by {num_classes} classes."
        )
    if num_outputs == num_classes:
        return trace

    grouped = trace.reshape(*trace.shape[:-1], num_classes, num_outputs // num_classes)
    if reduction == "sum":
        return grouped.sum(dim=-1)
    return grouped.mean(dim=-1)


def cross_entropy_scores(trace, mode, num_classes, population_reduction="mean"):
    """Return class scores used for accuracy reporting by a CE temporal mode."""
    class_trace = aggregate_population(trace, num_classes, population_reduction)
    if mode in ("temporal_mean", "per_timestep", "spike_rate"):
        return class_trace.mean(dim=0)
    if mode == "final_timestep":
        return class_trace[-1]
    if mode == "spike_count":
        return class_trace.sum(dim=0)
    raise ValueError(f"Unsupported CE mode: {mode}.")


def temporal_cross_entropy(
    trace,
    labels,
    mode,
    num_classes,
    population_reduction="mean",
):
    """Compute CE using a selected temporal and population aggregation method."""
    class_trace = aggregate_population(trace, num_classes, population_reduction)

    if mode == "per_timestep":
        num_steps, batch_size, _ = class_trace.shape
        repeated_labels = labels.unsqueeze(0).expand(num_steps, batch_size).reshape(-1)
        return F.cross_entropy(class_trace.reshape(-1, num_classes), repeated_labels)

    scores = cross_entropy_scores(
        class_trace,
        mode=mode,
        num_classes=num_classes,
        population_reduction=population_reduction,
    )
    return F.cross_entropy(scores, labels)
