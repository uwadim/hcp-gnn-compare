"""Тесты логистической регрессии на характеристиках графов."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from graph_features_logreg import (  # noqa: E402
    FEATURE_GROUPS,
    FeatureExtractor,
    LogRegConfig,
    Partition,
    build_cell,
    build_table,
    feature_set_slug,
    fit_predict_logreg,
    load_partition_csv,
    load_table,
    node_strengths,
    parse_feature_set,
    save_table,
    spatial_partition,
    strength_summary,
    strength_summary_names,
)


def _symmetric(values: dict[tuple[int, int], float], n: int) -> np.ndarray:
    matrix = np.zeros((n, n))
    for (i, j), v in values.items():
        matrix[i, j] = matrix[j, i] = v
    return matrix


def test_partition_validation() -> None:
    """Блок из одного узла и labels вне диапазона недопустимы."""
    with pytest.raises(ValueError, match="меньше 2"):
        Partition(labels=np.array([0, 0, 1]), names=("a", "b"))
    with pytest.raises(ValueError, match="диапазона"):
        Partition(labels=np.array([0, 0, 2, 2]), names=("a", "b"))
    part = Partition(labels=np.array([0, 0, 1, 1]), names=("a", "b"))
    assert part.n_nodes == 4 and part.n_blocks == 2
    assert part.tag == Partition(labels=np.array([0, 0, 1, 1]), names=("a", "b")).tag
    assert part.tag != Partition(labels=np.array([0, 1, 0, 1]), names=("a", "b")).tag


def test_block_means_known_values() -> None:
    """Средние внутри блоков и между ними по верхнему треугольнику."""
    part = Partition(labels=np.array([0, 0, 1, 1]), names=("a", "b"))
    matrix = _symmetric({(0, 1): 2.0, (2, 3): -4.0, (0, 2): 1.0, (0, 3): 3.0, (1, 2): 5.0, (1, 3): 7.0}, 4)
    extractor = FeatureExtractor(part)
    feats = extractor.extract(matrix)["block_means"]
    names = extractor.feature_names()["block_means"]
    assert names == ["a-a", "a-b", "b-b"]
    assert feats == pytest.approx([2.0, (1.0 + 3.0 + 5.0 + 7.0) / 4.0, -4.0])


def test_diagonal_is_ignored() -> None:
    """Диагональ не влияет на признаки."""
    part = Partition(labels=np.array([0, 0, 1, 1]), names=("a", "b"))
    matrix = _symmetric({(0, 1): 1.0, (2, 3): 2.0, (0, 2): 3.0}, 4)
    with_diag = matrix + np.diag([9.0, 9.0, 9.0, 9.0])
    extractor = FeatureExtractor(part)
    for group in FEATURE_GROUPS:
        assert extractor.extract(matrix)[group] == pytest.approx(extractor.extract(with_diag)[group])


def test_node_strengths_signed() -> None:
    """s, s+, s−, |s| для знакового графа."""
    matrix = _symmetric({(0, 1): 2.0, (0, 2): -3.0, (1, 2): 1.0}, 3)
    s = node_strengths(matrix)
    assert s["s_pos"] == pytest.approx([2.0, 3.0, 1.0])
    assert s["s_neg"] == pytest.approx([3.0, 0.0, 3.0])
    assert s["s"] == pytest.approx([-1.0, 3.0, -2.0])
    assert s["s_abs"] == pytest.approx([5.0, 3.0, 4.0])


def test_strength_summary_complete_graph() -> None:
    """Полный граф с единичными весами: все силы равны, std=0, энтропия=1."""
    n = 5
    matrix = np.ones((n, n)) - np.eye(n)
    summary = dict(zip(strength_summary_names(), strength_summary(node_strengths(matrix))))
    assert summary["s_mean"] == pytest.approx(n - 1)
    assert summary["s_std"] == 0.0 and summary["s_skew"] == 0.0 and summary["s_kurtosis"] == 0.0
    assert summary["s_neg_max"] == 0.0
    assert summary["s_abs_entropy"] == pytest.approx(1.0)
    assert len(summary) == 25 and np.isfinite(list(summary.values())).all()


def test_strength_summary_skewed_entropy_below_one() -> None:
    """Звёздный граф: энтропия |s| < 1, асимметрия > 0."""
    n = 6
    matrix = np.zeros((n, n))
    matrix[0, 1:] = matrix[1:, 0] = 1.0
    summary = dict(zip(strength_summary_names(), strength_summary(node_strengths(matrix))))
    assert 0.0 < summary["s_abs_entropy"] < 1.0
    assert summary["s_abs_skew"] > 0.0


def test_extract_validates_input() -> None:
    """Размер, NaN и Inf проверяются."""
    extractor = FeatureExtractor(Partition(labels=np.array([0, 0, 1, 1]), names=("a", "b")))
    with pytest.raises(ValueError, match="Ожидалась матрица"):
        extractor.extract(np.zeros((3, 3)))
    bad = np.zeros((4, 4))
    bad[0, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        extractor.extract(bad)


def _regions() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for hemisphere, sign in (("L", -1.0), ("R", 1.0)):
        for i in range(24):
            rows.append(
                (f"{hemisphere}_A{i}", sign * (10 + 40 * rng.random()), -90 + 130 * rng.random(), 60 * rng.random())
            )
    rows += [("thalamus_left", -10.0, -19.0, 6.0), ("thalamus_right", 11.0, -18.0, 6.0), ("brainStem", 0.0, -30.0, -32.0)]
    frame = pd.DataFrame(rows, columns=["label", "mni_x_mm", "mni_y_mm", "mni_z_mm"])
    frame.insert(0, "mmp_id", np.arange(1, len(frame) + 1))
    return frame


def test_spatial_partition() -> None:
    """k-means по полушариям, подкорка в одном блоке, имена спереди назад, детерминизм."""
    regions = _regions()
    part = spatial_partition(regions, n_clusters=3)
    assert part.names == ("L1", "L2", "L3", "R1", "R2", "R3", "SUB")
    sizes = np.bincount(part.labels)
    assert sizes[-1] == 3 and sizes[:6].sum() == 48
    left = part.labels[:24]
    assert set(left) <= {0, 1, 2}
    mean_y = [regions.loc[:23, "mni_y_mm"][left == k].mean() for k in range(3)]
    assert mean_y == sorted(mean_y, reverse=True)
    assert np.array_equal(part.labels, spatial_partition(regions, n_clusters=3).labels)
    with pytest.raises(ValueError, match="мало"):
        spatial_partition(regions, n_clusters=20)


def test_load_partition_csv(tmp_path: Path) -> None:
    """Разметка из CSV: порядок блоков по первому появлению, ошибки на пропусках."""
    regions = _regions().iloc[:6].reset_index(drop=True)
    csv = tmp_path / "net.csv"
    pd.DataFrame({"mmp_id": regions["mmp_id"], "network": list("ZZYYXX")}).to_csv(csv, index=False)
    part = load_partition_csv(csv, regions)
    assert part.names == ("Z", "Y", "X") and part.labels.tolist() == [0, 0, 1, 1, 2, 2]
    pd.DataFrame({"mmp_id": regions["mmp_id"][:5], "network": list("ZZYYX")}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="не размечены"):
        load_partition_csv(csv, regions)


def test_parse_feature_set() -> None:
    """Парсинг наборов признаков."""
    assert parse_feature_set("all") == FEATURE_GROUPS
    assert parse_feature_set("strength_summary+block_means") == ("block_means", "strength_summary")
    assert feature_set_slug("block_means+strength_summary") == "block_means_plus_strength_summary"
    with pytest.raises(ValueError, match="Неизвестные"):
        parse_feature_set("block_means+degree")


def _synthetic_xy(seed: int = 0, n_subjects: int = 60) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    x, y, groups = [], [], []
    for s in range(n_subjects):
        for label in (0, 1):
            x.append(np.r_[2.0 * (label - 0.5) + rng.normal(size=1), rng.normal(size=9)])
            y.append(label)
            groups.append(f"s{s}")
    return np.vstack(x), np.array(y, dtype=float), groups


def test_fit_predict_logreg_separable() -> None:
    """Информативный признак → высокий AUC; число фолдов и классы проверяются."""
    x, y, groups = _synthetic_xy(0)
    xt, yt, _ = _synthetic_xy(1)
    prob, best_c, cv_auc = fit_predict_logreg(x, y, groups, xt, LogRegConfig(c_grid=(0.01, 1.0)))
    assert roc_auc_score(yt, prob) > 0.85
    assert best_c in (0.01, 1.0) and cv_auc > 0.8
    with pytest.raises(ValueError, match="оба класса"):
        fit_predict_logreg(x, np.zeros_like(y), groups, xt, LogRegConfig())
    with pytest.raises(ValueError, match="cv_folds"):
        fit_predict_logreg(x[:6], y[:6], groups[:6], xt, LogRegConfig(cv_folds=5))
    with pytest.raises(ValueError, match="c_grid"):
        LogRegConfig(c_grid=(0.0,))


def _write_samples(tmp_path: Path, role: str, n_subjects: int, seed: int) -> list[SimpleNamespace]:
    """Граф: состояние 1 усиливает связи первого блока узлов."""
    rng = np.random.default_rng(seed)
    n = 12
    samples = []
    for s in range(n_subjects):
        for label in (0, 1):
            noise = rng.normal(scale=0.5, size=(n, n))
            matrix = (noise + noise.T) / 2.0
            matrix[:6, :6] += 1.5 * label
            np.fill_diagonal(matrix, 0.0)
            stem = f"{role}{s}_{label}"
            np.save(tmp_path / f"{stem}.npy", matrix)
            samples.append(
                SimpleNamespace(edge_path=tmp_path / f"{stem}.npy", stem=stem, subject=f"{role}{s}", label=float(label))
            )
    return samples


def test_end_to_end_synthetic_with_cache(tmp_path: Path) -> None:
    """build_table → build_cell даёт высокий AUC; кэш npz воспроизводит таблицу."""
    pool = _write_samples(tmp_path, "tr", 30, 0)
    test = _write_samples(tmp_path, "te", 20, 1)
    extractor = FeatureExtractor(Partition(labels=np.array([0] * 6 + [1] * 6), names=("A", "B")))
    table = build_table(pool, test, extractor, workers=2)
    assert table.groups["block_means"].shape == (len(pool) + len(test), 3)
    assert table.groups["strength_vector"].shape[1] == 12
    assert (table.meta["role"] == "train").sum() == len(pool)

    for spec in ("block_means", "strength_summary", "all"):
        cell = build_cell(table, spec, "v", "T", LogRegConfig(c_grid=(0.1, 1.0), cv_folds=3))
        auc = roc_auc_score(cell.test["label"], cell.test["prob"])
        assert auc > 0.9, spec
        assert cell.n_features == table.select(parse_feature_set(spec)).shape[1]
        assert list(cell.test.columns) == ["stem", "prob", "label"]

    path = tmp_path / "cache" / "t.npz"
    save_table(table, path)
    loaded = load_table(path)
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded.meta, table.meta)
    for group in FEATURE_GROUPS:
        np.testing.assert_array_equal(loaded.groups[group], table.groups[group])
    assert load_table(tmp_path / "cache" / "missing.npz") is None


def test_build_table_rejects_empty_and_bad_workers(tmp_path: Path) -> None:
    """Пустой test и workers<1 → ValueError."""
    pool = _write_samples(tmp_path, "tr", 2, 0)
    extractor = FeatureExtractor(Partition(labels=np.array([0] * 6 + [1] * 6), names=("A", "B")))
    with pytest.raises(ValueError, match="Пустой"):
        build_table(pool, [], extractor, workers=1)
    with pytest.raises(ValueError, match="workers"):
        build_table(pool, pool, extractor, workers=0)


def test_real_data_cell(data_root: Path, tmp_path: Path) -> None:
    """Реальные данные: 379 узлов, высокий AUC на ensemble_univariate/EMOTION."""
    from graph_features_logreg import collect_features, read_regions

    regions = read_regions(data_root)
    extractor = FeatureExtractor(spatial_partition(regions, 6))
    table = collect_features(data_root, "ensemble_univariate", "EMOTION", 0, extractor, 4, tmp_path)
    assert (table.meta["role"] == "test").sum() == 416
    assert table.groups["strength_vector"].shape[1] == 379
    cell = build_cell(table, "block_means+strength_summary", "ensemble_univariate", "EMOTION", LogRegConfig())
    assert roc_auc_score(cell.test["label"], cell.test["prob"]) > 0.95
    again = collect_features(data_root, "ensemble_univariate", "EMOTION", 0, extractor, 4, tmp_path)
    pd.testing.assert_frame_equal(table.meta, again.meta)
