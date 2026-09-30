"""Train from the command line."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.logging_conf import configure_logging
from src.models.train import run_training


def main() -> None:
    configure_logging()
    metadata = run_training()
    print(
        f"Trained {metadata['model_name']} "
        f"CV ROC-AUC {metadata['winner_cv_roc_auc']:.4f} "
        f"threshold {metadata['threshold']:.2f}"
    )


if __name__ == "__main__":
    main()
