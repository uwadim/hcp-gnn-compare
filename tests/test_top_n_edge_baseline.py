"""Тесты классификатора «средний вес top-n рёбер»."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mean_edge_baseline import mean_edge_weight  # noqa: E402
from top_n_edge_baseline import (  # noqa: E402
    best_n_by_train,
    build_n_grid,
    collect_scores,
    evaluate_cell,
    evaluate_cell_folds,
    fold_label,
    mean_curve,
    plot_curves,
    resolve_langs,
    top_n_means,
    ylim_for_curves,
)


def test_ylim_for_curves_fill_ratio() -> None:
    """Размах данных занимает ровно fill_ratio высоты оси; вырождение даёт min_span."""
    ymin, ymax = ylim_for_curves(np.array([0.8, 0.9]), np.array([0.85, 1.0]), fill_ratio=0.75)
    assert ymax - ymin == pytest.approx((1.0 - 0.8) / 0.75)
    assert ymin == pytest.approx(0.8 - (ymax - ymin - 0.2) / 2)
    dmin, dmax = ylim_for_curves(np.array([0.5]), np.array([0.5]), fill_ratio=0.75, min_span=0.01)
    assert dmax - dmin == pytest.approx(0.01 / 0.75)
    with pytest.raises(ValueError):
        ylim_for_curves(np.array([0.0]), np.array([1.0]), fill_ratio=0.0)


def _matrix() -> np.ndarray:
    """Рёбра верхнего треугольника: [3, -5, 1, 4, -2, 6] → |w|: 6,5,4,3,2,1."""
    m = np.zeros((4, 4))
    values = [3.0, -5.0, 1.0, 4.0, -2.0, 6.0]
    for (i, j), v in zip(zip(*np.triu_indices(4, 1)), values):
        m[i, j] = m[j, i] = v
    return m


def test_top_n_means_known_values() -> None:
    """Среднее знаковых весов n рёбер с наибольшим |w|; n=все совпадает со средним."""
    got = top_n_means(_matrix(), [1, 2, 3, 6])
    # По |w|: 6, -5, 4, 3, -2, 1.
    assert got == pytest.approx([6.0, 0.5, 5 / 3, mean_edge_weight(_matrix())])
    assert got[-1] == pytest.approx(7 / 6)


def test_top_n_means_rejects_bad_n() -> None:
    """n вне [1, число рёбер] и неквадратная матрица → ValueError."""
    with pytest.raises(ValueError, match="n должен"):
        top_n_means(_matrix(), [0])
    with pytest.raises(ValueError, match="n должен"):
        top_n_means(_matrix(), [7])
    with pytest.raises(ValueError, match="квадратная"):
        top_n_means(np.zeros((3, 4)), [1])


def test_build_n_grid() -> None:
    """Сетка отсортирована, без повторов и значений > n_edges, заканчивается n_edges."""
    assert build_n_grid(100, [50, 5, 5, 1000, 1]) == [1, 5, 50, 100]
    assert build_n_grid(10, [10]) == [10]
    with pytest.raises(ValueError):
        build_n_grid(10, [0])
    with pytest.raises(ValueError):
        build_n_grid(0, [1])


def _synthetic_frame(n_subjects: int = 60, seed: int = 0) -> pd.DataFrame:
    """top1 разделяет классы, top5 — шум."""
    rng = np.random.default_rng(seed)
    rows = []
    for role in ("train", "test"):
        for s in range(n_subjects):
            for label in (0.0, 1.0):
                rows.append(
                    {
                        "stem": f"{role[0]}{s}_RL_{int(label)}",
                        "subject": f"{role[0]}{s}",
                        "label": label,
                        "role": role,
                        "top1": (label - 0.5) * 4.0 + 0.1 * rng.normal(),
                        "top5": rng.normal(),
                    }
                )
    return pd.DataFrame(rows)


def test_evaluate_cell_and_best_n() -> None:
    """Информативный n даёт AUC≈1, шумовой ≈0.5; выбор n идёт по train."""
    table, comparison = evaluate_cell(_synthetic_frame(), "v", "WM", [1, 5], 200, 0)
    assert list(table["n"]) == [1, 5]
    by_n = table.set_index("n")
    assert by_n.loc[1, "roc_auc"] > 0.99
    assert 0.3 < by_n.loc[5, "roc_auc"] < 0.7
    assert (by_n["ci_low"] <= by_n["roc_auc"]).all() and (by_n["roc_auc"] <= by_n["ci_high"]).all()
    assert set(comparison.replicates) == {"1", "5"}
    best = best_n_by_train(table)
    assert int(best.iloc[0]["best_n"]) == 1


def test_evaluate_cell_folds_pools_with_per_fold_sign() -> None:
    """Знак подбирается по train своего fold; pooled AUC по всем fold, один набор субъектов."""
    first = _synthetic_frame(seed=1)
    second = _synthetic_frame(seed=2)
    second["top1"] = -second["top1"]
    second["subject"] = "b" + second["subject"]
    second["stem"] = "b" + second["stem"]
    table, comparison = evaluate_cell_folds([first, second], "v", "WM", [1, 5], 200, 0)
    by_n = table.set_index("n")
    assert by_n.loc[1, "roc_auc"] > 0.99
    assert by_n.loc[1, "sign"] == 0
    assert 0.3 < by_n.loc[5, "roc_auc"] < 0.7
    assert len(comparison.subjects) == 2 * 60
    with pytest.raises(ValueError, match="frames пуст"):
        evaluate_cell_folds([], "v", "WM", [1], 10, 0)


def test_fold_label() -> None:
    """Один fold — номер, несколько — перечисление с пометкой pooled."""
    assert fold_label([2]) == "2"
    assert fold_label([0, 1, 2, 3]) == "0, 1, 2, 3 (pooled)"


def test_mean_curve_and_plot(tmp_path: Path) -> None:
    """Среднее по задачам и график создаётся."""
    tables, comparisons = [], []
    for seed, task in enumerate(("WM", "MOTOR")):
        table, comparison = evaluate_cell(_synthetic_frame(seed=seed), "v", task, [1, 5], 200, 0)
        tables.append(table)
        comparisons.append(comparison)
    mean = mean_curve("v", comparisons)
    assert set(mean["task"]) == {"MEAN"} and list(mean["n"]) == [1, 5]
    paths = plot_curves(pd.concat(tables), mean, tmp_path, outer_fold=0)
    assert {p.name for p in paths} == {
        "auc_vs_n_v.png", "auc_vs_n_v.svg", "auc_vs_n_mean.png", "auc_vs_n_mean.svg",
    }
    assert all(p.is_file() and p.stat().st_size > 1000 for p in paths)

    paths_en = plot_curves(pd.concat(tables), mean, tmp_path, outer_fold=0, lang="en")
    assert {p.name for p in paths_en} == {
        "auc_vs_n_v_en.png", "auc_vs_n_v_en.svg", "auc_vs_n_mean_en.png", "auc_vs_n_mean_en.svg",
    }
    # svg.fonttype=none: подписи остаются текстом, их можно проверить в SVG
    svg_en = (tmp_path / "auc_vs_n_v_en.svg").read_text(encoding="utf-8")
    assert "Test ROC-AUC" in svg_en and "number of edges" in svg_en and "Task" in svg_en
    svg_ru = (tmp_path / "auc_vs_n_v.svg").read_text(encoding="utf-8")
    assert "ROC-AUC на test" in svg_ru
    with pytest.raises(ValueError, match="язык"):
        plot_curves(pd.concat(tables), mean, tmp_path, outer_fold=0, lang="de")


def test_resolve_langs() -> None:
    """both раскрывается в оба языка, неизвестный код → ValueError."""
    assert resolve_langs("both") == ("ru", "en")
    assert resolve_langs("en") == ("en",)
    with pytest.raises(ValueError, match="язык"):
        resolve_langs("fr")


def test_real_data_cell(data_root: Path, tmp_path: Path) -> None:
    """Реальные данные: n=все совпадает с mean-edge; кэш воспроизводим."""
    ns = [1, 100, 71631]
    frame = collect_scores(data_root, "ensemble_univariate", "EMOTION", 0, ns, 4, tmp_path)
    assert (frame["role"] == "test").sum() == 416
    table, _ = evaluate_cell(frame, "ensemble_univariate", "EMOTION", ns, 200, 0)
    assert table.set_index("n").loc[71631, "roc_auc"] > 0.99
    again = collect_scores(data_root, "ensemble_univariate", "EMOTION", 0, ns, 4, tmp_path)
    pd.testing.assert_frame_equal(frame, again)
