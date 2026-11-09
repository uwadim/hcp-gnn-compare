"""Тесты классификатора «средний вес ребра»."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mean_edge_baseline import (  # noqa: E402
    build_cell,
    collect_scores,
    fit_sign,
    is_degenerate,
    mean_edge_weight,
    run_baseline,
)


def test_mean_edge_weight_upper_triangle() -> None:
    """Среднее по верхнему треугольнику без диагонали."""
    m = np.array([[9.0, 1.0, 2.0], [1.0, 9.0, 3.0], [2.0, 3.0, 9.0]])
    assert mean_edge_weight(m) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="квадратная"):
        mean_edge_weight(np.zeros((3, 4)))


def test_fit_sign_flips_negative_relation() -> None:
    """Обратная связь скора с меткой даёт знак −1 и AUC > 0.5 после разворота."""
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    assert fit_sign(y, np.array([-3, -2, -1, 1, 2, 3.0])) == (1, pytest.approx(1.0))
    sign, auc = fit_sign(y, np.array([3, 2, 1, -1, -2, -3.0]))
    assert (sign, auc) == (-1, pytest.approx(1.0))
    with pytest.raises(ValueError, match="один класс"):
        fit_sign(np.ones(4), np.arange(4.0))


def test_degenerate_score_detected() -> None:
    """Константный (шум float) скор распознаётся как вырожденный."""
    rng = np.random.default_rng(0)
    noise = rng.normal(scale=1e-17, size=50)
    y = (rng.random(50) > 0.5).astype(float)
    assert is_degenerate(noise)
    assert not is_degenerate(np.linspace(-1, 1, 50))
    assert np.isnan(fit_sign(y, noise)[1])


def test_build_cell_applies_train_sign() -> None:
    """Знак подбирается на train и применяется к test."""
    frame = pd.DataFrame(
        {
            "stem": [f"{i}_RL_a" for i in range(8)],
            "subject": [str(i) for i in range(8)],
            "label": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "role": ["train"] * 4 + ["test"] * 4,
            "score": [3.0, -3.0, 2.0, -2.0, 1.0, -1.0, 4.0, -4.0],
        }
    )
    cell = build_cell(frame, "ensemble_full", "WM")
    assert cell.sign == -1
    assert not cell.degenerate
    assert list(cell.test["prob"]) == [-1.0, 1.0, -4.0, 4.0]
    assert list(cell.test["label"]) == [0.0, 1.0, 0.0, 1.0]


def test_collect_scores_real_data(data_root: Path, tmp_path: Path) -> None:
    """Реальные данные: ensemble EMOTION разделяется почти идеально, кэш воспроизводим."""
    frame = collect_scores(data_root, "ensemble_univariate", "EMOTION", 0, 4, tmp_path)
    assert set(frame["role"]) == {"train", "test"}
    assert (frame["role"] == "test").sum() == 416
    cell = build_cell(frame, "ensemble_univariate", "EMOTION")
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(cell.test["label"], cell.test["prob"]) > 0.99
    again = collect_scores(data_root, "ensemble_univariate", "EMOTION", 0, 4, tmp_path)
    pd.testing.assert_frame_equal(frame, again)


def test_correlation_variant_is_degenerate(data_root: Path, tmp_path: Path) -> None:
    """Стандартизованные correlation-графы: средний вес ≡ 0."""
    frame = collect_scores(data_root, "correlation", "EMOTION", 0, 4, tmp_path)
    assert build_cell(frame, "correlation", "EMOTION").degenerate


def test_run_baseline_single_report_for_several_folds(data_root: Path, tmp_path: Path) -> None:
    """Несколько fold → один report.md, pooled по 416 субъектам, диагностика по каждому fold."""
    run_baseline(data_root, tmp_path, tasks=("EMOTION",), outer_folds=(0, 1), n_bootstrap=100)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "outer fold: 0, 1" in report and "субъектов в pooled: 416" in report
    diagnostics = pd.read_csv(tmp_path / "cell_diagnostics.csv")
    assert set(diagnostics["fold"]) == {0, 1}
    pooled = pd.read_csv(tmp_path / "auc_pooled.csv")
    assert {"task", "variant", "roc_auc"} <= set(pooled.columns)
    assert (tmp_path / "auc_per_fold.csv").is_file()
