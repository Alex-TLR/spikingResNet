#!/usr/bin/env python3
"""Compare test accuracy across SNN loss configurations.

The script loads one YAML file as the base experiment configuration and creates
an independent configuration for every requested loss variant. For each
variant, it derives the final-weight path under ``weights/spike/loss/``.
Existing weights are evaluated directly;
when the expected file is absent, the repository's normal ``training`` entry
point is called first and the resulting artifact is verified before testing.

The default sweep covers the historical loss names ``rate_loss``,
``count_loss``, ``cross_entropy``, and ``mse_count_loss``. Passing ``--all-ce``
also generates every valid combination of CE source and temporal mode:
pre-LIF logits, output membranes, or spikes combined with their supported
temporal aggregations. These explicit CE variants receive unique weight tags so
checkpoints trained with different semantics cannot overwrite or reuse one
another accidentally.

The experiment arrays ``resnet_models``, ``expansions``, and ``seeds`` define a
Cartesian product. Every loss variant runs at every point in that grid. After
each run, the script atomically rewrites a JSON file containing every per-seed
result and a TXT report containing mean and population-standard-deviation
accuracy across completed seeds for each variant, model, and expansion. By
default, errors are recorded and the sweep continues; ``--fail-fast`` persists
the failed run before stopping. On restart, a compatible existing JSON file is
loaded and completed loss/model/expansion/seed entries are skipped.

Examples:
    Run the four historical losses::

        python tools/loss_accuracy_sweep.py \
            --config experiments/loss_accuracy/cifar10.yaml

    Run all explicit CE variants with mean population reduction::

        python tools/loss_accuracy_sweep.py \
            --config experiments/loss_accuracy/cifar10.yaml \
            --all-ce --population-reduction mean

The module keeps orchestration separate from model implementation. The
``run_loss_sweep`` function accepts injected training and testing callables so
its control flow can be unit-tested without downloading datasets or launching
full training.
"""

import argparse
import copy
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import load_config_from_yaml
from utils.experiment_paths import weight_path
from utils.snn_loss import resolve_ce_options


HISTORICAL_LOSSES = (
    "rate_loss",
    "count_loss",
    "cross_entropy",
    "mse_count_loss",
)

CE_MODES_BY_SOURCE = {
    "logits": ("temporal_mean", "per_timestep", "final_timestep"),
    "membranes": ("temporal_mean", "per_timestep", "final_timestep"),
    "spikes": (
        "temporal_mean",
        "per_timestep",
        "final_timestep",
        "spike_rate",
        "spike_count",
    ),
}


def utc_now():
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def build_loss_variants(base_config, losses, all_ce=False, population_reduction="mean"):
    """Create independent configurations for requested loss experiments.

    Args:
        base_config: Loaded experiment configuration used as the common model,
            dataset, optimizer, and runtime setup.
        losses: Iterable of historical loss names to include in the sweep.
        all_ce: When true, append every valid explicit CE source/mode pair.
        population_reduction: Population aggregation used by generated CE
            variants; either ``mean`` or ``sum``.

    Returns:
        A list of ``(variant_name, config)`` pairs. Every configuration is a
        deep copy, so mutating one variant cannot affect another. Duplicate
        names are collapsed while retaining their final configuration.

    Notes:
        Historical CE aliases are converted to explicit source/mode/reduction
        names. This prevents ``rate_loss`` from duplicating an exhaustive
        per-timestep spike CE run under a misleading weight filename.
    """
    variants = []
    for loss_name in losses:
        config = copy.deepcopy(base_config)
        if loss_name == "rate_loss":
            config.loss = "cross_entropy"
            config.ce_source = "spikes"
            config.ce_mode = "per_timestep"
            config.population_reduction = population_reduction
            variant_name = f"ce_spikes_per_timestep_{population_reduction}"
        elif loss_name == "count_loss":
            config.loss = "cross_entropy"
            config.ce_source = "spikes"
            config.ce_mode = "spike_count"
            config.population_reduction = "sum"
            variant_name = "ce_spikes_spike_count_sum"
        elif loss_name == "cross_entropy":
            config.ce_source, config.ce_mode, config.population_reduction = resolve_ce_options(
                config.ce_source,
                config.ce_mode,
                config.population_reduction,
                config.fit,
            )
            variant_name = (
                f"ce_{config.ce_source}_{config.ce_mode}_"
                f"{config.population_reduction}"
            )
        else:
            if config.fit != "spike":
                raise ValueError(
                    f"{loss_name} requires training.fit='spike'; received "
                    f"training.fit='{config.fit}'. Use a spike-fit YAML."
                )
            config.loss = loss_name
            config.ce_source = None
            config.ce_mode = None
            config.population_reduction = None
            variant_name = loss_name
        config.weight_tag = f"{variant_name}_fit_{config.fit}"
        variants.append((variant_name, config))

    if all_ce:
        for source, modes in CE_MODES_BY_SOURCE.items():
            for mode in modes:
                name = f"ce_{source}_{mode}_{population_reduction}"
                config = copy.deepcopy(base_config)
                config.loss = "cross_entropy"
                config.ce_source = source
                config.ce_mode = mode
                config.population_reduction = population_reduction
                config.weight_tag = f"{name}_fit_{config.fit}"
                variants.append((name, config))

    unique_variants = {}
    for name, config in variants:
        unique_variants[name] = config
    return list(unique_variants.items())


def default_results_path(config):
    """Build the default JSON path from identity-defining experiment fields.

    Args:
        config: Experiment configuration containing dataset, timestep,
            augmentation, and experiment-case values.

    Returns:
        A path below ``results/``. The loss is omitted because one JSON document
        stores all loss variants in the sweep.
    """
    case = str(config.case).replace("/", "_")
    filename = (
        f"loss_accuracy_{config.model_type}_{config.dataset_ID}"
        f"_{case}_T{config.num_time_steps_train}_A{config.auto_aug}.json"
    )
    return Path("results") / filename


def default_report_path(results_path):
    """Return the TXT report path corresponding to a JSON results path.

    Args:
        results_path: Configured or generated JSON output path.

    Returns:
        The same path with a ``.txt`` suffix.
    """
    return Path(results_path).with_suffix(".txt")


def numeric(value):
    """Convert a scalar tensor or numeric value to a finite JSON float.

    Args:
        value: Python numeric value or scalar object exposing ``item()``.

    Returns:
        The value converted to ``float``.

    Raises:
        TypeError: If the value cannot be converted to a float.
        ValueError: If the converted value is NaN or infinite.
    """
    if hasattr(value, "item"):
        value = value.item()
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Accuracy must be finite, received {value}.")
    return value


def sweep_values(config, plural_name, singular_name):
    """Resolve a non-empty, duplicate-free sweep axis from configuration."""
    values = getattr(config, plural_name, None)
    if values is None:
        values = [getattr(config, singular_name)]
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{plural_name} must be a non-empty list or tuple.")
    return list(dict.fromkeys(values))


def summarize_results(payload):
    """Aggregate completed per-seed runs by variant, model, and expansion."""
    expected_seed_count = len(payload["experiment"]["seeds"])
    summary = {}
    for variant_name, model_results in payload["results"].items():
        summary[variant_name] = {}
        for model_name, expansion_results in model_results.items():
            summary[variant_name][model_name] = {}
            for expansion_name, seed_results in expansion_results.items():
                completed = [
                    entry
                    for entry in seed_results.values()
                    if entry.get("status") == "completed"
                ]
                spike_values = [
                    entry["spike_accuracy"]
                    for entry in completed
                    if entry.get("spike_accuracy") is not None
                ]
                membrane_values = [
                    entry["membrane_accuracy"]
                    for entry in completed
                    if entry.get("membrane_accuracy") is not None
                ]
                configured_values = [
                    entry["configured_accuracy"]
                    for entry in completed
                    if entry.get("configured_accuracy") is not None
                ]

                def aggregate(values):
                    if not values:
                        return None, None
                    return statistics.fmean(values), statistics.pstdev(values)

                spike_mean, spike_std = aggregate(spike_values)
                membrane_mean, membrane_std = aggregate(membrane_values)
                configured_mean, configured_std = aggregate(configured_values)
                completed_count = len(completed)
                status = (
                    "completed"
                    if completed_count == expected_seed_count
                    else "partial" if completed_count else "failed"
                )
                summary[variant_name][model_name][expansion_name] = {
                    "status": status,
                    "completed_seeds": completed_count,
                    "total_seeds": expected_seed_count,
                    "spike_accuracy_mean": spike_mean,
                    "spike_accuracy_std": spike_std,
                    "membrane_accuracy_mean": membrane_mean,
                    "membrane_accuracy_std": membrane_std,
                    "configured_accuracy_mean": configured_mean,
                    "configured_accuracy_std": configured_std,
                }
    payload["summary"] = summary
    return summary


def format_mean_std(mean, std):
    """Format an optional mean and standard deviation for the TXT report."""
    if mean is None:
        return "-"
    return f"{mean:.2f} +/- {std:.2f}"


def write_json_atomic(path, payload):
    """Atomically serialize a sweep payload as formatted JSON.

    The payload is first written to a sibling ``.tmp`` file and then moved over
    the destination. This prevents an interruption during serialization from
    leaving a partially written results document.

    Args:
        path: Destination path for the JSON document.
        payload: JSON-serializable mapping containing experiment metadata and
            per-variant results.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary_path.replace(path)


def load_or_create_payload(results_path, experiment, overwrite=False):
    """Load resumable sweep data or create a new compatible payload."""
    results_path = Path(results_path)
    if overwrite or not results_path.exists():
        return {
            "schema_version": 2,
            "generated_at": utc_now(),
            "experiment": experiment,
            "results": {},
            "summary": {},
        }

    try:
        with results_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot resume invalid results file {results_path}: {exc}") from exc

    if payload.get("schema_version") != 2:
        raise ValueError(
            f"Cannot resume {results_path}: expected schema version 2, "
            f"found {payload.get('schema_version')}."
        )
    stored_experiment = payload.get("experiment")
    sweep_axes = ("resnet_models", "expansions", "seeds")
    fixed_metadata = {
        key: value for key, value in experiment.items() if key not in sweep_axes
    }
    stored_fixed_metadata = {
        key: value
        for key, value in (stored_experiment or {}).items()
        if key not in sweep_axes
    }
    if stored_fixed_metadata != fixed_metadata:
        raise ValueError(
            f"Cannot resume {results_path}: stored experiment metadata does not "
            "match the current configuration. Use --overwrite or a different "
            "--output path."
        )
    merged_experiment = dict(experiment)
    for axis in sweep_axes:
        stored_values = stored_experiment.get(axis, [])
        current_values = experiment.get(axis, [])
        merged_experiment[axis] = list(
            dict.fromkeys([*stored_values, *current_values])
        )
    payload["experiment"] = merged_experiment
    if not isinstance(payload.get("results"), dict):
        raise ValueError(f"Cannot resume {results_path}: results must be an object.")

    payload.setdefault("summary", {})
    return payload


def write_text_report(path, payload):
    """Write grouped mean/std accuracy and per-seed failures atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    experiment = payload["experiment"]
    results = payload["results"]
    summary = payload.get("summary") or summarize_results(payload)

    with temporary_path.open("w", encoding="utf-8") as stream:
        stream.write("Loss Accuracy Sweep Report\n")
        stream.write("=" * 120 + "\n")
        stream.write(f"Generated: {payload['generated_at']}\n")
        stream.write(f"Dataset: {experiment['dataset']}\n")
        stream.write(f"Model type: {experiment['model_type']}\n")
        stream.write(f"Case: {experiment['case']}\n")
        stream.write(
            f"Timesteps (train/extract): {experiment['time_steps_train']}/"
            f"{experiment['time_steps_extract']}\n"
        )
        stream.write(f"ResNet models: {experiment['resnet_models']}\n")
        stream.write(f"Expansions: {experiment['expansions']}\n")
        stream.write(f"Seeds: {experiment['seeds']}\n")
        stream.write(
            f"Epochs: {experiment['epochs']} | AutoAugment: "
            f"{experiment['auto_aug']}\n\n"
        )
        stream.write(
            f"{'Variant':<42} {'Model':>6} {'Expansion':>9} {'Seeds':>7} "
            f"{'Status':<10} {'Configured mean/std':>20} "
            f"{'Spike mean/std':>18} {'Membrane mean/std':>20}\n"
        )
        stream.write("-" * 142 + "\n")

        for variant_name, model_results in summary.items():
            for model_name, expansion_results in model_results.items():
                for expansion_name, entry in expansion_results.items():
                    spike_text = format_mean_std(
                        entry["spike_accuracy_mean"], entry["spike_accuracy_std"]
                    )
                    membrane_text = format_mean_std(
                        entry["membrane_accuracy_mean"],
                        entry["membrane_accuracy_std"],
                    )
                    configured_text = format_mean_std(
                        entry["configured_accuracy_mean"],
                        entry["configured_accuracy_std"],
                    )
                    seed_text = f"{entry['completed_seeds']}/{entry['total_seeds']}"
                    stream.write(
                        f"{variant_name:<42} {model_name:>6} {expansion_name:>9} "
                        f"{seed_text:>7} {entry['status']:<10} "
                        f"{configured_text:>20} {spike_text:>18} "
                        f"{membrane_text:>20}\n"
                    )

        failures = []
        for variant_name, model_results in results.items():
            for model_name, expansion_results in model_results.items():
                for expansion_name, seed_results in expansion_results.items():
                    for seed_name, entry in seed_results.items():
                        if entry.get("error"):
                            run_name = (
                                f"{variant_name} R{model_name} "
                                f"E{expansion_name} S{seed_name}"
                            )
                            failures.append((run_name, entry["error"]))
        if failures:
            stream.write("\nFailures\n")
            stream.write("-" * 120 + "\n")
            for run_name, error in failures:
                stream.write(f"{run_name}: {error}\n")

    temporary_path.replace(path)


def run_loss_sweep(
    base_config,
    variants,
    train_fn,
    test_fn,
    results_path,
    report_path=None,
    weights_root="weights",
    fail_fast=False,
    overwrite=False,
):
    """Run every loss/model/expansion/seed combination and persist each run.

    Args:
        base_config: Common configuration whose plural experiment fields define
            the model, expansion, and seed axes.
        variants: Sequence of ``(variant_name, config)`` pairs, normally from
            :func:`build_loss_variants`.
        train_fn: Callable receiving one variant configuration. It must create
            the final artifact returned by :func:`utils.experiment_paths.weight_path`.
        test_fn: Callable receiving one variant configuration. For spiking
            models it returns ``(spike_accuracy, membrane_accuracy)`` or those
            values plus configured accuracy; for conventional models it returns
            membrane/classification accuracy.
        results_path: Destination JSON file. Parent directories are created.
        report_path: Destination TXT report. When omitted, it uses the JSON
            filename with a ``.txt`` suffix.
        weights_root: Root directory below which the sweep uses
            ``<model_type>/loss`` for existing and newly trained weights. The
            production default therefore resolves to ``weights/spike/loss``
            for SNN configurations; tests inject a temporary root.
        fail_fast: If true, persist the current failure and then raise an error.
        overwrite: If true, discard existing result metadata and start a fresh
            results payload. Existing model weights are still reused.

    Returns:
        A payload containing nested per-seed results and grouped summaries.

    Each result records whether training was triggered. Training is skipped
    only when the exact variant-specific final weight file already exists. A
    training callback that returns without producing that file is treated as a
    failure, preventing evaluation of stale or unrelated weights.
    """
    results_path = Path(results_path)
    report_path = Path(report_path) if report_path else default_report_path(results_path)
    resnet_models = sweep_values(base_config, "resnet_models", "resnet_model")
    expansions = sweep_values(base_config, "expansions", "expansion")
    seeds = sweep_values(base_config, "seeds", "seed")
    experiment = {
        "dataset": base_config.dataset_ID,
        "model_type": base_config.model_type,
        "resnet_models": resnet_models,
        "case": str(base_config.case),
        "time_steps_train": base_config.num_time_steps_train,
        "time_steps_extract": base_config.num_time_steps_extract,
        "expansions": expansions,
        "auto_aug": base_config.auto_aug,
        "seeds": seeds,
        "epochs": base_config.epochs,
    }
    payload = load_or_create_payload(results_path, experiment, overwrite=overwrite)
    summarize_results(payload)
    write_json_atomic(results_path, payload)
    write_text_report(report_path, payload)

    total_runs = len(variants) * len(resnet_models) * len(expansions) * len(seeds)
    run_index = 0
    print(
        f"Sweep grid: {len(variants)} losses x {len(resnet_models)} models x "
        f"{len(expansions)} expansions x {len(seeds)} seeds = {total_runs} runs"
    )

    for resnet_model in resnet_models:
        for expansion in expansions:
            for seed in seeds:
                for variant_name, variant_config in variants:
                    run_index += 1
                    config = copy.deepcopy(variant_config)
                    config.resnet_model = resnet_model
                    config.expansion = expansion
                    config.seed = seed
                    config.weight_directory = (
                        Path(weights_root) / config.model_type / "loss"
                    )
                    path = weight_path(config)
                    existing_entry = (
                        payload["results"]
                        .get(variant_name, {})
                        .get(str(resnet_model), {})
                        .get(str(expansion), {})
                        .get(str(seed))
                    )
                    if (
                        isinstance(existing_entry, dict)
                        and existing_entry.get("status") == "completed"
                        and existing_entry.get("weight_file") == str(path)
                    ):
                        print(
                            f"\n[{run_index}/{total_runs}] Skipping completed "
                            f"{variant_name} | ResNet{resnet_model} | "
                            f"E={expansion} | S={seed}"
                        )
                        continue
                    entry = run_single_configuration(
                        config,
                        variant_name,
                        path,
                        train_fn,
                        test_fn,
                        run_index,
                        total_runs,
                    )
                    variant_results = payload["results"].setdefault(variant_name, {})
                    model_results = variant_results.setdefault(str(resnet_model), {})
                    expansion_results = model_results.setdefault(str(expansion), {})
                    expansion_results[str(seed)] = entry
                    payload["generated_at"] = utc_now()
                    summarize_results(payload)
                    write_json_atomic(results_path, payload)
                    write_text_report(report_path, payload)

                    if fail_fast and entry["status"] == "failed":
                        raise RuntimeError(entry["error"])

    return payload


def run_single_configuration(
    config,
    variant_name,
    path,
    train_fn,
    test_fn,
    run_index,
    total_runs,
):
    """Train if needed and evaluate one grid point."""
    trained = False
    print(
        f"\n[{run_index}/{total_runs}] {variant_name} | "
        f"ResNet{config.resnet_model} | E={config.expansion} | S={config.seed}"
    )
    print(f"Weights: {path}")
    entry = {
        "loss": config.loss,
        "fit": config.fit,
        "ce_source": getattr(config, "ce_source", None),
        "ce_mode": getattr(config, "ce_mode", None),
        "population_reduction": getattr(config, "population_reduction", None),
        "resnet_model": config.resnet_model,
        "expansion": config.expansion,
        "seed": config.seed,
        "weight_file": str(path),
        "training_triggered": False,
        "status": "pending",
    }

    try:
        if not path.exists():
            print("Weights not found; starting training.")
            path.parent.mkdir(parents=True, exist_ok=True)
            train_fn(config)
            trained = True
            if not path.exists():
                raise RuntimeError(f"Training finished without creating {path}.")
        else:
            print("Weights found; skipping training.")

        accuracy = test_fn(config)
        if config.model_type == "spike":
            if len(accuracy) == 3:
                spike_accuracy, membrane_accuracy, configured_accuracy = accuracy
                entry["configured_accuracy"] = numeric(configured_accuracy)
            else:
                spike_accuracy, membrane_accuracy = accuracy
                entry["configured_accuracy"] = None
            entry["spike_accuracy"] = numeric(spike_accuracy)
            entry["membrane_accuracy"] = numeric(membrane_accuracy)
        else:
            entry["spike_accuracy"] = None
            entry["membrane_accuracy"] = numeric(accuracy)
            entry["configured_accuracy"] = numeric(accuracy)
        entry["status"] = "completed"
    except Exception as exc:
        entry["status"] = "failed"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        print(entry["error"])

    entry["training_triggered"] = trained
    entry["evaluated_at"] = utc_now()
    return entry


def get_parser():
    """Build the command-line parser for historical and exhaustive CE sweeps."""
    parser = argparse.ArgumentParser(
        description="Compare test accuracy across SNN loss functions."
    )
    parser.add_argument("--config", required=True, help="Base YAML experiment config")
    parser.add_argument(
        "--losses",
        nargs="+",
        choices=HISTORICAL_LOSSES,
        default=None,
        help=(
            "Historical loss names to evaluate. Defaults to all historical "
            "losses unless --all-ce is used, in which case none are added."
        ),
    )
    parser.add_argument(
        "--all-ce",
        action="store_true",
        help="Also evaluate every valid explicit CE source/mode combination",
    )
    parser.add_argument(
        "--population-reduction",
        choices=("mean", "sum"),
        default="mean",
        help="Population reduction used by --all-ce variants",
    )
    parser.add_argument("--output", help="JSON output path; defaults under results/")
    parser.add_argument(
        "--report",
        help="TXT report path; defaults to the JSON filename with a .txt suffix",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing JSON/TXT sweep results instead of resuming them",
    )
    return parser


def main():
    """Load CLI options and execute the sweep with repository train/test APIs."""
    args = get_parser().parse_args()
    config = load_config_from_yaml(args.config)
    losses = args.losses
    if losses is None:
        losses = [] if args.all_ce else list(HISTORICAL_LOSSES)
    variants = build_loss_variants(
        config,
        losses,
        all_ce=args.all_ce,
        population_reduction=args.population_reduction,
    )
    results_path = Path(args.output) if args.output else default_results_path(config)
    report_path = Path(args.report) if args.report else default_report_path(results_path)

    from test import test_accuracy
    from train import training

    run_loss_sweep(
        config,
        variants,
        train_fn=training,
        test_fn=lambda sweep_config: test_accuracy(
            sweep_config, return_configured_accuracy=True
        ),
        results_path=results_path,
        report_path=report_path,
        fail_fast=args.fail_fast,
        overwrite=args.overwrite,
    )
    print(f"\nJSON data saved to: {results_path}")
    print(f"TXT report saved to: {report_path}")


if __name__ == "__main__":
    main()