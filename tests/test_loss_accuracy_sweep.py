"""Unit tests for loss-accuracy sweep orchestration and artifact paths."""

import json
import tempfile
import unittest
from pathlib import Path

from config import ExperimentConfig
from tools.loss_accuracy_sweep import (
    build_loss_variants,
    load_or_create_payload,
    run_loss_sweep,
)
from utils.experiment_paths import checkpoint_path, weight_path


class LossAccuracySweepTest(unittest.TestCase):
    """Verify variant generation, weight reuse, training, and JSON output."""

    COUNT_CE_NAME = "ce_spikes_spike_count_sum"

    def make_config(self):
        """Create a compact deterministic configuration for path tests."""
        config = ExperimentConfig()
        config.case = "unit"
        config.resnet_model = 4
        config.expansion = 2
        config.num_time_steps_train = 3
        config.num_time_steps_extract = 3
        config.auto_aug = False
        config.seed = 7
        config.epochs = 1
        return config

    def test_weight_paths_preserve_legacy_name_and_support_tags(self):
        """Artifact paths retain old names and allow explicit CE tags."""
        config = self.make_config()
        config.loss = "cross_entropy"
        expected = (
            "weights/spike/expunit/resnet4_weights_CIFAR10"
            "_T_3_E_2_L_cross_entropy_A_False_S_7.pth"
        )
        self.assertEqual(str(weight_path(config)), expected)

        config.weight_tag = "ce_logits_temporal_mean_mean"
        self.assertIn("_L_ce_logits_temporal_mean_mean_", str(weight_path(config)))
        self.assertTrue(str(checkpoint_path(config)).endswith("_checkpoint.pth"))

        config.weight_directory = "weights/spike/loss"
        self.assertEqual(weight_path(config).parent, Path("weights/spike/loss"))
        self.assertEqual(checkpoint_path(config).parent, Path("weights/spike/loss"))

    def test_overwrite_replaces_incompatible_results_payload(self):
        """Explicit overwrite starts fresh despite stored metadata mismatch."""
        old_experiment = {"case": "old"}
        new_experiment = {"case": "new"}

        with tempfile.TemporaryDirectory() as directory:
            results_path = Path(directory) / "accuracy.json"
            results_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "generated_at": "old",
                        "experiment": old_experiment,
                        "results": {"old": {}},
                        "summary": {},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "metadata does not match"):
                load_or_create_payload(results_path, new_experiment)
            payload = load_or_create_payload(
                results_path, new_experiment, overwrite=True
            )

        self.assertEqual(payload["experiment"], new_experiment)
        self.assertEqual(payload["results"], {})

    def test_resume_merges_sweep_axes_without_eliminating_results(self):
        """Changing the requested grid preserves stored axis values and results."""
        stored_experiment = {
            "case": "unit",
            "resnet_models": [4, 10, 18],
            "expansions": [1],
            "seeds": [42],
        }
        current_experiment = {
            "case": "unit",
            "resnet_models": [10, 18],
            "expansions": [1, 2],
            "seeds": [42, 7],
        }
        stored_results = {"variant": {"4": {"1": {"42": {}}}}}

        with tempfile.TemporaryDirectory() as directory:
            results_path = Path(directory) / "accuracy.json"
            results_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "generated_at": "old",
                        "experiment": stored_experiment,
                        "results": stored_results,
                        "summary": {},
                    }
                ),
                encoding="utf-8",
            )
            payload = load_or_create_payload(results_path, current_experiment)

        self.assertEqual(payload["experiment"]["resnet_models"], [4, 10, 18])
        self.assertEqual(payload["experiment"]["expansions"], [1, 2])
        self.assertEqual(payload["experiment"]["seeds"], [42, 7])
        self.assertEqual(payload["results"], stored_results)

    def test_all_ce_builds_every_valid_source_mode(self):
        """The exhaustive CE sweep keeps one YAML-selected training fit."""
        config = self.make_config()
        config.fit = "membrane"
        variants = build_loss_variants(config, [], all_ce=True)

        self.assertEqual(len(variants), 11)
        variants_by_name = dict(variants)
        names = set(variants_by_name)
        self.assertIn("ce_logits_temporal_mean_mean", names)
        self.assertIn("ce_membranes_final_timestep_mean", names)
        self.assertIn("ce_spikes_spike_count_mean", names)
        self.assertEqual(
            variants_by_name["ce_logits_temporal_mean_mean"].fit,
            "membrane",
        )
        self.assertEqual(
            variants_by_name["ce_membranes_final_timestep_mean"].fit,
            "membrane",
        )
        self.assertEqual(
            variants_by_name["ce_spikes_spike_count_mean"].fit,
            "membrane",
        )
        self.assertTrue(
            all(variant.fit == "membrane" for variant in variants_by_name.values())
        )

    def test_historical_ce_aliases_use_explicit_names(self):
        """Legacy rate/count inputs produce canonical, non-duplicate CE tags."""
        config = self.make_config()
        config.ce_source = "logits"
        config.ce_mode = "temporal_mean"
        config.population_reduction = "mean"
        config.fit = "membrane"
        variants = build_loss_variants(
            config, ["rate_loss", "count_loss", "cross_entropy"], all_ce=True
        )
        names = [name for name, _config in variants]

        self.assertEqual(len(names), 12)
        self.assertNotIn("rate_loss", names)
        self.assertNotIn("count_loss", names)
        self.assertEqual(names.count("ce_spikes_per_timestep_mean"), 1)
        self.assertIn(self.COUNT_CE_NAME, names)
        logits_config = dict(variants)["ce_logits_temporal_mean_mean"]
        self.assertEqual(
            logits_config.weight_tag,
            "ce_logits_temporal_mean_mean_fit_membrane",
        )

    def test_sweep_trains_missing_weights_and_writes_json_and_txt(self):
        """The sweep reuses weights and writes machine and human outputs."""
        config = self.make_config()
        variants = build_loss_variants(config, ["count_loss", "mse_count_loss"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "weights"
            results_path = Path(directory) / "results" / "accuracy.json"
            for _name, variant_config in variants:
                variant_config.weight_directory = root / "spike" / "loss"
            existing_path = weight_path(variants[0][1])
            existing_path.parent.mkdir(parents=True)
            existing_path.write_bytes(b"existing")
            trained = []
            tested = []

            def fake_train(variant_config):
                """Record training and create the expected fake artifact."""
                trained.append(variant_config.weight_tag)
                path = weight_path(variant_config)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"trained")

            def fake_test(variant_config):
                """Record evaluation and return deterministic accuracies."""
                tested.append(variant_config.weight_tag)
                return 91.25, 92.5

            payload = run_loss_sweep(
                config,
                variants,
                train_fn=fake_train,
                test_fn=fake_test,
                results_path=results_path,
                weights_root=root,
            )

            persisted = json.loads(results_path.read_text(encoding="utf-8"))
            report = results_path.with_suffix(".txt").read_text(encoding="utf-8")

        self.assertEqual(trained, ["mse_count_loss_fit_spike"])
        self.assertEqual(
            tested,
            ["ce_spikes_spike_count_sum_fit_spike", "mse_count_loss_fit_spike"],
        )
        self.assertEqual(payload["results"], persisted["results"])
        count_entry = persisted["results"][self.COUNT_CE_NAME]["4"]["2"]["7"]
        mse_entry = persisted["results"]["mse_count_loss"]["4"]["2"]["7"]
        self.assertFalse(count_entry["training_triggered"])
        self.assertTrue(mse_entry["training_triggered"])
        self.assertEqual(count_entry["spike_accuracy"], 91.25)
        self.assertEqual(count_entry["membrane_accuracy"], 92.5)
        self.assertEqual(
            persisted["summary"][self.COUNT_CE_NAME]["4"]["2"]["completed_seeds"],
            1,
        )
        self.assertIn("Loss Accuracy Sweep Report", report)
        self.assertIn(self.COUNT_CE_NAME, report)
        self.assertIn("91.25 +/- 0.00", report)
        self.assertIn("92.50 +/- 0.00", report)

    def test_missing_training_artifact_is_recorded_as_failure(self):
        """A training run that creates no weights is persisted as failed."""
        config = self.make_config()
        variants = build_loss_variants(config, ["count_loss"])

        with tempfile.TemporaryDirectory() as directory:
            results_path = Path(directory) / "accuracy.json"
            payload = run_loss_sweep(
                config,
                variants,
                train_fn=lambda _config: None,
                test_fn=lambda _config: self.fail("test must not run"),
                results_path=results_path,
                weights_root=Path(directory) / "weights",
            )
            persisted = json.loads(results_path.read_text(encoding="utf-8"))
            report = results_path.with_suffix(".txt").read_text(encoding="utf-8")

        entry = payload["results"][self.COUNT_CE_NAME]["4"]["2"]["7"]
        self.assertEqual(entry["status"], "failed")
        self.assertIn("Training finished without creating", entry["error"])
        self.assertEqual(
            entry,
            persisted["results"][self.COUNT_CE_NAME]["4"]["2"]["7"],
        )
        self.assertEqual(
            persisted["summary"][self.COUNT_CE_NAME]["4"]["2"]["status"],
            "failed",
        )
        self.assertIn("Failures", report)
        self.assertIn("Training finished without creating", report)

    def test_sweep_runs_cartesian_grid_and_averages_seeds(self):
        """Every model/expansion/seed point is retained and summarized."""
        config = self.make_config()
        config.resnet_models = [4, 10]
        config.expansions = [1, 5]
        config.seeds = [7, 11]
        variants = build_loss_variants(config, ["count_loss"])

        with tempfile.TemporaryDirectory() as directory:
            results_path = Path(directory) / "accuracy.json"
            trained = []
            tested = []

            def fake_train(variant_config):
                """Create one uniquely named artifact for every grid point."""
                identity = (
                    variant_config.resnet_model,
                    variant_config.expansion,
                    variant_config.seed,
                )
                trained.append(identity)
                path = weight_path(variant_config)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"trained")

            def fake_test(variant_config):
                """Return seed-dependent values with predictable statistics."""
                tested.append(
                    (
                        variant_config.resnet_model,
                        variant_config.expansion,
                        variant_config.seed,
                    )
                )
                return (
                    float(variant_config.seed),
                    float(variant_config.seed + 1),
                    float(variant_config.seed + 2),
                )

            payload = run_loss_sweep(
                config,
                variants,
                train_fn=fake_train,
                test_fn=fake_test,
                results_path=results_path,
                weights_root=Path(directory) / "weights",
            )
            persisted = json.loads(results_path.read_text(encoding="utf-8"))
            report = results_path.with_suffix(".txt").read_text(encoding="utf-8")

        expected_points = {
            (model, expansion, seed)
            for model in (4, 10)
            for expansion in (1, 5)
            for seed in (7, 11)
        }
        self.assertEqual(set(trained), expected_points)
        self.assertEqual(set(tested), expected_points)
        self.assertEqual(payload["results"], persisted["results"])
        self.assertEqual(
            set(persisted["results"][self.COUNT_CE_NAME]["10"]["5"]),
            {"7", "11"},
        )
        summary = persisted["summary"][self.COUNT_CE_NAME]["10"]["5"]
        self.assertEqual(summary["completed_seeds"], 2)
        self.assertEqual(summary["total_seeds"], 2)
        self.assertEqual(summary["spike_accuracy_mean"], 9.0)
        self.assertEqual(summary["spike_accuracy_std"], 2.0)
        self.assertEqual(summary["membrane_accuracy_mean"], 10.0)
        self.assertEqual(summary["membrane_accuracy_std"], 2.0)
        self.assertEqual(summary["configured_accuracy_mean"], 11.0)
        self.assertEqual(summary["configured_accuracy_std"], 2.0)
        self.assertIn("9.00 +/- 2.00", report)
        self.assertIn("11.00 +/- 2.00", report)
        self.assertEqual(report.count(self.COUNT_CE_NAME), 4)

    def test_sweep_persists_each_result_and_resumes_unfinished_seed(self):
        """An interruption preserves completed seeds and restart skips them."""
        config = self.make_config()
        config.seeds = [7, 11]
        variants = build_loss_variants(config, ["count_loss"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "weights"
            results_path = Path(directory) / "accuracy.json"

            def fake_train(variant_config):
                """Create the expected artifact before evaluation."""
                path = weight_path(variant_config)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"trained")

            def interrupt_second_seed(variant_config):
                """Simulate Ctrl+C while the second seed is being tested."""
                if variant_config.seed == 11:
                    raise KeyboardInterrupt
                return 71.0, 72.0, 73.0

            with self.assertRaises(KeyboardInterrupt):
                run_loss_sweep(
                    config,
                    variants,
                    train_fn=fake_train,
                    test_fn=interrupt_second_seed,
                    results_path=results_path,
                    weights_root=root,
                )

            interrupted = json.loads(results_path.read_text(encoding="utf-8"))
            seed_results = interrupted["results"][self.COUNT_CE_NAME]["4"]["2"]
            self.assertEqual(set(seed_results), {"7"})
            self.assertEqual(seed_results["7"]["configured_accuracy"], 73.0)

            resumed_tests = []

            def resumed_test(variant_config):
                """Record the seed evaluated after loading the checkpoint."""
                resumed_tests.append(variant_config.seed)
                return 81.0, 82.0, 83.0

            payload = run_loss_sweep(
                config,
                variants,
                train_fn=lambda _config: self.fail("weights should already exist"),
                test_fn=resumed_test,
                results_path=results_path,
                weights_root=root,
            )

        self.assertEqual(resumed_tests, [11])
        seed_results = payload["results"][self.COUNT_CE_NAME]["4"]["2"]
        self.assertEqual(set(seed_results), {"7", "11"})
        self.assertEqual(seed_results["7"]["configured_accuracy"], 73.0)
        self.assertEqual(seed_results["11"]["configured_accuracy"], 83.0)
        self.assertEqual(
            payload["summary"][self.COUNT_CE_NAME]["4"]["2"]["status"],
            "completed",
        )


if __name__ == "__main__":
    unittest.main()
