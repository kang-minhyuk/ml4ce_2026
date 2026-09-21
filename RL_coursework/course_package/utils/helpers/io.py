"""Helper functions for reinforcement learning algorithms."""

import csv
import importlib.util
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from time import time
from types import ModuleType

import numpy as np
import torch


# todo: extend to log global progress
class CSVLogger:
    def __init__(self, filename: str):
        self.filename = filename
        write_header = not Path(filename).exists()
        self.file = open(filename, "a", newline="")
        self.writer = csv.writer(self.file)

        if write_header:
            self.writer.writerow([
                "step",
                "episode",
                "agent_id",
                "train_return",
                "eval_return",
                "policy_loss",
                "value_loss",
                "entropy",
                "timestamp",
            ])

    def log(
        self,
        step,
        episode,
        agent_id,
        train_return,
        eval_return,
        policy_loss,
        value_loss,
        entropy,
    ):
        self.writer.writerow([
            step,
            episode,
            agent_id,
            train_return,
            eval_return,
            policy_loss,
            value_loss,
            entropy,
            time(),
        ])

    def close(self):
        self.file.close()


def set_seed(seed: int):
    """ "Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)


def outputs_root() -> Path:
    """Folder where trained policies, plots and submissions are written.

    Defaults to ``./outputs``; the Docker image sets ``COURSE_OUTPUT_DIR`` to the
    exposed outputs volume."""
    return Path(os.environ.get("COURSE_OUTPUT_DIR", "outputs"))


def setup_model_saving(algorithm: str) -> Path:
    """
    Simple helper to generate a filename for saving model weights.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    folder = outputs_root() / "trained_models" / f"{algorithm}_{timestamp}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def validate_module(module: ModuleType) -> None:
    """Check that an algorithm module defines ``train`` and ``AlgorithmConfig``."""
    module_name = module.__name__
    if not hasattr(module, "train"):
        raise AttributeError(f"{module_name} must define train()")
    if not hasattr(module, "AlgorithmConfig"):
        raise AttributeError(f"{module_name} must define AlgorithmConfig")


def export_submission(
    dev_path: str | Path, group_id: str, submissions_dir: str | Path | None = None
) -> Path:
    """Validate your algorithm file and copy it to ``<outputs>/submissions/<group_id>.py``."""
    dev_path = Path(dev_path)
    spec = importlib.util.spec_from_file_location(dev_path.stem, dev_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    validate_module(module)

    submissions_dir = (
        Path(submissions_dir) if submissions_dir else outputs_root() / "submissions"
    )
    submissions_dir.mkdir(parents=True, exist_ok=True)
    target = submissions_dir / f"{group_id}.py"
    shutil.copy(dev_path, target)
    print(f"Submission saved to {target}")
    return target


def save_training_plots(output_dir: str | Path, *, title: str) -> None:
    """Save the learning-curve figures of a completed run next to its CSV logs."""
    import matplotlib.pyplot as plt

    from course_package.utils.plots.metrics import plot_learning_curves

    output_dir = Path(output_dir)
    figure = plot_learning_curves(output_dir, title=title, show=False)
    figure.savefig(output_dir / "learning_curve.png", bbox_inches="tight")
    plt.close(figure)
