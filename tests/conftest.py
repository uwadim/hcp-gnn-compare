"""Pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

# Архив лежит рядом с проектом: ../HPC_data_ref/hcp_mmp_mean_ablation_graphs.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    _PROJECT_ROOT / ".." / "HPC_data_ref" / "hcp_mmp_mean_ablation_graphs"
).resolve()


@pytest.fixture(scope="session")
def data_root() -> Path:
    """
    Data root.

    Returns
    -------
    Path
        Корень архива HCP, если он существует.
    """
    if not DATA_ROOT.is_dir():
        pytest.skip(f"Нет данных: {DATA_ROOT}")
    return DATA_ROOT
