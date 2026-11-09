"""
Stage 2 entry: сравнение модели на test с bootstrap CI.

Ожидает лучшие гиперпараметры SkipGCN (после stage 1) через CLI/overrides
или значения из ``conf/model/skipgcn.yaml`` / ``cfmsr.yaml``.
"""

import hydra
from omegaconf import DictConfig

from train import run_training


@hydra.main(version_base=None, config_path="conf", config_name="config_compare")
def main(cfg: DictConfig) -> float:
    """
    Hydra entry for stage-2 comparison.

    Примеры::

        python main_compare.py variant=correlation task=EMOTION model=skipgcn
        python main_compare.py variant=ensemble_full task=WM model=cfmsr \\
            training.skipgcn.hidden_channels=24 training.lr=5e-4
    """
    return float(run_training(cfg))


if __name__ == "__main__":
    main()
