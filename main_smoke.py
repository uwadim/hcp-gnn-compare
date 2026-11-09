"""
Базовый прогон с дефолтными гиперпараметрами (проверка сходимости).
"""

import hydra
from omegaconf import DictConfig

from train import run_training


@hydra.main(version_base=None, config_path="conf", config_name="config_smoke")
def main(cfg: DictConfig) -> float:
    """
    Hydra entry for smoke / convergence runs.

    Примеры::

        python main_smoke.py model=skipgcn variant=correlation task=WM
        python main_smoke.py -m
    """
    return float(run_training(cfg))


if __name__ == "__main__":
    main()
