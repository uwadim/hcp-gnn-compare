"""Тесты bootstrap ROC-AUC CI."""

from __future__ import annotations

import numpy as np

from bootstrap_metrics import bootstrap_roc_auc_ci


def test_bootstrap_ci_perfect_separation() -> None:
    """Идеальное разделение: AUC≈1, CI высокий."""
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    s = np.array([0.1, 0.2, 0.15, 0.25, 0.8, 0.9, 0.85, 0.95], dtype=float)
    auc, lo, hi = bootstrap_roc_auc_ci(y, s, n_bootstrap=200, seed=0)
    assert auc > 0.95
    assert lo <= auc <= hi
    assert hi > 0.8


def test_bootstrap_ci_single_class() -> None:
    """Один класс → NaN."""
    y = np.ones(10)
    s = np.linspace(0, 1, 10)
    auc, lo, hi = bootstrap_roc_auc_ci(y, s, n_bootstrap=50, seed=1)
    assert np.isnan(auc)
    assert np.isnan(lo)
    assert np.isnan(hi)
