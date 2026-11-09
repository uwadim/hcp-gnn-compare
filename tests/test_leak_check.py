"""Тесты пермутации меток и аудита ensemble."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dataset import (  # noqa: E402
    GraphSample,
    HCPDataModule,
    shuffle_sample_labels,
)
from audit_ensemble_construction import (  # noqa: E402
    audit_fold,
    edge_label_correlation,
    render_report,
)


def _sample(stem: str, label: float) -> GraphSample:
    return GraphSample(
        stem=stem,
        subject=stem.split("_", maxsplit=1)[0],
        state="a",
        label=label,
        edge_path=Path(f"{stem}.npy"),
        vertex_path=None,
        outer_fold=0,
        inner_fold=0,
        split_role="train_pool",
    )


def test_shuffle_sample_labels_preserves_multiset() -> None:
    """Пермутация сохраняет мультимножество меток и меняет порядок."""
    samples = [_sample(f"{i}_RL_a", float(i % 2)) for i in range(20)]
    before = [s.label for s in samples]
    shuffle_sample_labels(samples, seed=0)
    after = [s.label for s in samples]
    assert sorted(before) == sorted(after)
    assert before != after
    shuffle_sample_labels(samples, seed=0)
    # Повтор с тем же seed на уже перемешанном списке — другая перестановка
    # исходных after; баланс тот же.
    assert sorted(after) == sorted(s.label for s in samples)


def test_shuffle_empty_raises() -> None:
    """Пустой список → ValueError."""
    with pytest.raises(ValueError, match="пустого"):
        shuffle_sample_labels([], seed=0)


def _base_cfg(data_root: Path, **extra: object) -> dict:
    cfg = {
        "root": str(data_root),
        "variant": "ensemble_univariate",
        "task": "MOTOR",
        "node_features": "ones",
        "drop_percentile": 90.0,
        "edge_scaler": "none",
        "batch_size": 2,
        "num_workers": 0,
        "outer_fold": 0,
        "inner_val_fold": 0,
        "shuffle_labels_seed": 1,
    }
    cfg.update(extra)
    return cfg


def test_datamodule_shuffle_all_changes_test_labels(data_root: Path) -> None:
    """scope=all меняет метки и на test; баланс классов на test сохраняется."""
    dm_true = HCPDataModule(OmegaConf.create(_base_cfg(data_root, shuffle_labels=False)))
    dm_true.setup_single_split()
    true_test = np.asarray(dm_true.test_dataset.labels)

    dm = HCPDataModule(
        OmegaConf.create(
            _base_cfg(
                data_root,
                shuffle_labels=True,
                shuffle_labels_scope="all",
                shuffle_labels_seed=1,
            )
        )
    )
    dm.setup_single_split()
    shuffled = np.asarray(dm.test_dataset.labels)
    # Баланс сохраняется на pool+test вместе, не обязательно на одном test.
    true_all = np.concatenate(
        [
            np.asarray(dm_true.train_dataset.labels),
            np.asarray(dm_true.val_dataset.labels),
            true_test,
        ]
    )
    shuffled_all = np.concatenate(
        [
            np.asarray(dm.train_dataset.labels),
            np.asarray(dm.val_dataset.labels),
            shuffled,
        ]
    )
    assert int(true_all.sum()) == int(shuffled_all.sum())
    assert not np.array_equal(true_test, shuffled)


def test_datamodule_shuffle_train_keeps_test_labels(data_root: Path) -> None:
    """scope=train не трогает test; train pool меняется."""
    dm_true = HCPDataModule(OmegaConf.create(_base_cfg(data_root, shuffle_labels=False)))
    dm_true.setup_outer(dm_true.splits[0])
    true_test = [s.label for s in dm_true._test_samples]
    true_pool = [s.label for s in dm_true._pool_samples]

    dm = HCPDataModule(
        OmegaConf.create(
            _base_cfg(
                data_root,
                shuffle_labels=True,
                shuffle_labels_scope="train",
                shuffle_labels_seed=2,
            )
        )
    )
    dm.setup_outer(dm.splits[0])
    assert [s.label for s in dm._test_samples] == true_test
    shuffled_pool = [s.label for s in dm._pool_samples]
    assert sorted(shuffled_pool) == sorted(true_pool)
    assert shuffled_pool != true_pool


def test_audit_fold_motor_univariate(data_root: Path) -> None:
    """fold0: субъекты совпадают со splits, corr(edge,label) высокая."""
    row = audit_fold(data_root, "ensemble_univariate", "MOTOR", outer_fold=0)
    assert row["train_match_splits"] is True
    assert row["test_match_splits"] is True
    assert row["train_test_disjoint"] is True
    assert row["inner_pairwise_disjoint"] is True
    assert float(row["corr_mean_edge_label_test"]) > 0.8
    report = render_report(data_root, [row])
    assert "supervised" in report.lower() or "P(state2)" in report


def test_edge_label_correlation_known(tmp_path: Path) -> None:
    """Синтетика: corr = 1 при идеальном разделении средних."""
    labels = {"a": 0.0, "b": 1.0}
    for stem, y in labels.items():
        matrix = np.full((4, 4), 1.0 if y else -1.0, dtype=np.float32)
        np.fill_diagonal(matrix, 0.0)
        np.save(tmp_path / f"{stem}.npy", matrix)
    corr, m0, m1, n = edge_label_correlation(tmp_path, labels)
    assert n == 2
    assert corr == pytest.approx(1.0)
    assert m0 < 0 < m1
