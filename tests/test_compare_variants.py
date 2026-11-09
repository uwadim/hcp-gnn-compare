"""Тесты кластерного/парного bootstrap и сравнения вариантов."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bootstrap_metrics import (  # noqa: E402
    bh_adjust,
    bootstrap_auc_replicates,
    holm_adjust,
    percentile_ci,
    romano_wolf_adjust,
    studentized_p_values,
    subject_from_stem,
)
from compare_variants import (  # noqa: E402
    MEAN_TASK,
    align_predictions,
    auc_table,
    compare_task,
    mean_over_tasks,
    pairwise_table,
)


def _preds(n_subjects: int, noise: float, seed: int) -> pd.DataFrame:
    """Синтетика: 2 run × 2 condition на субъекта, сбалансированные метки."""
    rng = np.random.default_rng(seed)
    rows = []
    for subj in range(n_subjects):
        for run in ("LR", "RL"):
            for cond, label in (("a", 0), ("b", 1)):
                rows.append(
                    {
                        "stem": f"{100000 + subj}_{run}_{cond}",
                        "label": float(label),
                        "prob": float(label + noise * rng.normal()),
                    }
                )
    return pd.DataFrame(rows)


def test_subject_from_stem() -> None:
    """Субъект — префикс stem до первого подчёркивания."""
    assert subject_from_stem("100206_RL_0bk") == "100206"
    with pytest.raises(ValueError):
        subject_from_stem("_RL_0bk")


def test_cluster_bootstrap_keeps_subjects_whole() -> None:
    """Кластерные реплики одинаковы для одинаковых субъектов и CI содержит точку."""
    df = _preds(30, noise=1.0, seed=0)
    groups = [subject_from_stem(s) for s in df["stem"]]
    point, reps = bootstrap_auc_replicates(
        df["label"], {"m": df["prob"]}, groups=groups, n_bootstrap=300, seed=1,
    )
    low, high = percentile_ci(reps["m"])
    assert low <= point["m"] <= high
    _, reps_again = bootstrap_auc_replicates(
        df["label"], {"m": df["prob"]}, groups=groups, n_bootstrap=300, seed=1,
    )
    np.testing.assert_allclose(reps["m"], reps_again["m"])


def test_paired_bootstrap_detects_better_variant() -> None:
    """Явно лучший вариант: ΔAUC > 0, CI не содержит 0, p мал."""
    good = _preds(40, noise=0.3, seed=0)
    bad = _preds(40, noise=2.0, seed=1)
    result = compare_task("WM", {"bad": bad, "good": good}, n_bootstrap=500, seed=0)
    pairs = pairwise_table([result], reference=None)
    row = pairs.iloc[0]
    assert (row["variant_a"], row["variant_b"]) == ("bad", "good")
    assert row["delta"] > 0
    assert row["ci_low"] > 0
    assert row["p_value"] < 0.05


def test_identical_variants_not_significant() -> None:
    """Одинаковые скоры: ΔAUC = 0, p = 1."""
    df = _preds(20, noise=1.0, seed=3)
    result = compare_task("WM", {"a": df, "b": df.copy()}, n_bootstrap=200, seed=0)
    row = pairwise_table([result], reference=None).iloc[0]
    assert row["delta"] == pytest.approx(0.0)
    assert row["p_value"] == pytest.approx(1.0)
    assert row["p_romano_wolf"] == pytest.approx(1.0)


def test_align_rejects_mismatched_samples() -> None:
    """Разные наборы test-образцов или метки → ошибка."""
    df = _preds(5, noise=1.0, seed=0)
    with pytest.raises(ValueError, match="набор"):
        align_predictions({"a": df, "b": df.iloc[1:]})
    flipped = df.copy()
    flipped.loc[0, "label"] = 1.0 - flipped.loc[0, "label"]
    with pytest.raises(ValueError, match="Метки"):
        align_predictions({"a": df, "b": flipped})


def test_mean_over_tasks_and_tables() -> None:
    """MEAN — среднее точечных AUC; Holm считается отдельно для MEAN."""
    res = [
        compare_task(t, {"a": _preds(20, 1.0, i), "b": _preds(20, 0.5, i + 10)}, 200, 0)
        for i, t in enumerate(("WM", "MOTOR"))
    ]
    mean = mean_over_tasks(res)
    assert mean.task == MEAN_TASK
    assert mean.point["a"] == pytest.approx(np.mean([r.point["a"] for r in res]))
    aucs = auc_table([*res, mean])
    assert set(aucs["rank"]) <= {1, 2}
    pairs = pairwise_table([*res, mean], reference=None)
    assert len(pairs) == 3
    assert set(pairs["family"]) == {"primary"}
    assert (pairs["p_romano_wolf"] >= pairs["p_value"] - 1e-12).all()
    assert (pairs["p_holm"] >= pairs["p_value"] - 1e-12).all()


def test_families_split_by_reference() -> None:
    """Пары с эталоном — primary (эталон всегда a), остальные — secondary."""
    preds = {v: _preds(20, 1.0, i) for i, v in enumerate(("ref", "x", "y"))}
    pairs = pairwise_table([compare_task("WM", preds, 200, 0)], reference="ref")
    primary = pairs[pairs["family"] == "primary"]
    assert set(primary["variant_a"]) == {"ref"}
    assert len(primary) == 2
    assert len(pairs[pairs["family"] == "secondary"]) == 1
    with pytest.raises(ValueError, match="reference"):
        pairwise_table([compare_task("WM", preds, 200, 0)], reference="missing")


def test_romano_wolf_no_penalty_for_identical_hypotheses() -> None:
    """Полностью зависимые гипотезы: RW = p одиночной проверки, Holm штрафует ×m."""
    rng = np.random.default_rng(0)
    col = rng.normal(loc=0.3, scale=0.15, size=5000)
    est = np.array([0.3])
    single = romano_wolf_adjust(est, col[:, None])[0]
    multi = romano_wolf_adjust(np.repeat(est, 5), np.repeat(col[:, None], 5, axis=1))
    assert multi == pytest.approx([single] * 5)
    assert holm_adjust([single] * 5)[0] == pytest.approx(min(1.0, 5 * single))


def test_romano_wolf_close_to_holm_for_independent() -> None:
    """Независимые гипотезы: RW близок к Holm/Šidák, не меньше одиночного p."""
    rng = np.random.default_rng(1)
    m, n_boot = 6, 20000
    est = np.array([0.5, 0.3, 0.2, 0.1, 0.05, 0.0]) * 0.1
    reps = est + rng.normal(scale=0.1, size=(n_boot, m)) * 0.1
    rw = np.array(romano_wolf_adjust(est, reps))
    single = np.array([romano_wolf_adjust(est[[j]], reps[:, [j]])[0] for j in range(m)])
    holm = np.array(holm_adjust(single.tolist()))
    assert np.all(rw >= single - 1e-12)
    assert np.all(rw <= holm + 0.02)
    order = np.argsort(-np.abs(est))
    assert np.all(np.diff(rw[order]) >= -1e-12)


def test_romano_wolf_rejects_bad_shapes() -> None:
    """Несовпадение форм → ValueError."""
    with pytest.raises(ValueError, match="форма"):
        romano_wolf_adjust([0.1, 0.2], np.zeros((10, 3)))


def test_bh_adjust() -> None:
    """BH: известный пример, монотонность и nan."""
    q = bh_adjust([0.01, 0.04, 0.03, 0.2, float("nan")])
    assert q[:4] == pytest.approx([0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.2])
    assert np.isnan(q[4])


def test_rank_auc_matches_sklearn() -> None:
    """Ранговый AUC совпадает со sklearn, в том числе при связях."""
    from sklearn.metrics import roc_auc_score

    from bootstrap_metrics import _rank_auc

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    s = np.round(rng.normal(size=200) + y, 1)
    assert _rank_auc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_holm_and_p_value() -> None:
    """Holm монотонен и не меньше сырых p; p-value в (0, 1]."""
    adjusted = holm_adjust([0.01, 0.04, 0.03, float("nan")])
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[2] == pytest.approx(0.06)
    assert adjusted[1] == pytest.approx(0.06)
    assert np.isnan(adjusted[3])
    p = studentized_p_values([0.1, 0.0], np.array([[0.12, 0.0], [0.08, 0.0], [0.11, 0.0]]))
    assert 0.0 < p[0] < 1.0
    assert p[1] == pytest.approx(1.0)
