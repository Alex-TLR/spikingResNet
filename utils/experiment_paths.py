"""Canonical paths for trained model artifacts."""

from pathlib import Path


def loss_artifact_tag(config):
    """Return the filename-safe loss identifier for an experiment."""
    return str(getattr(config, "weight_tag", None) or config.loss)


def weight_path(config, root="weights"):
    """Build the final model-weight path used by training and evaluation."""
    directory_override = getattr(config, "weight_directory", None)
    directory = (
        Path(directory_override)
        if directory_override is not None
        else Path(root) / config.model_type / f"exp{config.case}"
    )
    tag = loss_artifact_tag(config)
    if config.model_type == "conv":
        filename = (
            f"resnet{config.resnet_model}_weights_{config.dataset_ID}"
            f"_T_{config.num_time_steps_train}_E_{config.expansion}"
            f"_L_{tag}_A_{config.auto_aug}_S_{config.seed}.pth"
        )
    elif config.model_type == "spike":
        filename = (
            f"resnet{config.resnet_model}_weights_{config.dataset_ID}"
            f"_T_{config.num_time_steps_train}_E_{config.expansion}"
            f"_L_{tag}_A_{config.auto_aug}_S_{config.seed}.pth"
        )
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")
    return directory / filename


def checkpoint_path(config, root="weights"):
    """Build the resumable checkpoint path corresponding to final weights."""
    final_path = weight_path(config, root=root)
    return final_path.with_name(f"{final_path.stem}_checkpoint.pth")