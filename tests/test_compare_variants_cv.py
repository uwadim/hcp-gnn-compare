"""Тесты многофолдового сравнения вариантов и плана stage2_cv."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import compare_variants_cv as cvc  # noqa: E402
from batch_runner import (  # noqa: E402
    OUTER_FOLDS,
    SKIPGCN_DEFAULT_HP,
    TASKS,
    SKIPGCN_CORR_AVG_BEST_HP,
    VARIANTS,
    build_stage2_cv_jobs,
)

SUBJECTS_PER_FOLD = 20
NOISES = {"correlation": 1.5, "ensemble_correlation": 1.0, "ensemble_univariate": 0.6, "ensemble_full": 0.4}


def _fold_subjects(fold: int) -> List[str]:
    return [str(100000 + fold * 1000 + i) for i in range(SUBJECTS_PER_FOLD)]


def _fold_preds(fold: int, noise: float, seed: int) -> pd.DataFrame:
    """Синтетика fold: 2 run × 2 condition на субъекта."""
    rng = np.random.default_rng(seed + fold)
    rows = []
    for subject in _fold_subjects(fold):
        for run in ("LR", "RL"):
            for cond, label in (("a", 0), ("b", 1)):
                rows.append(
                    {
                        "stem": f"{subject}_{run}_{cond}",
                        "label": float(label),
                        "prob": float(label + noise * rng.normal()),
                    }
                )
    return pd.DataFrame(rows)


def _preds_by_fold(folds: List[int], task_seed: int = 0) -> Dict[int, Dict[str, pd.DataFrame]]:
    return {
        fold: {v: _fold_preds(fold, NOISES[v], seed=task_seed + 10 * i) for i, v in enumerate(VARIANTS)}
        for fold in folds
    }


@pytest.fixture()
def splits_path(tmp_path: Path) -> Path:
    """splits.json на 4 fold без пересечений train/test."""
    everyone = [s for k in OUTER_FOLDS for s in _fold_subjects(k)]
    payload = [
        {
            "fold": k,
            "train": [s for s in everyone if s not in _fold_subjects(k)],
            "test": _fold_subjects(k),
            "inner": [],
        }
        for k in OUTER_FOLDS
    ]
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_pool_predictions_concatenates_folds() -> None:
    """Pooled содержит все образцы всех fold без дубликатов stem."""
    pooled = cvc.pool_predictions(_preds_by_fold([0, 1, 2]))
    assert set(pooled) == set(VARIANTS)
    for frame in pooled.values():
        assert len(frame) == 3 * SUBJECTS_PER_FOLD * 4
        assert not frame["stem"].duplicated().any()


def test_pool_predictions_rejects_different_variants() -> None:
    """Разный набор вариантов между fold — ошибка."""
    preds = _preds_by_fold([0, 1])
    del preds[1]["ensemble_full"]
    with pytest.raises(ValueError, match="Набор вариантов"):
        cvc.pool_predictions(preds)


def test_validate_fold_subjects(splits_path: Path) -> None:
    """Состав субъектов fold сверяется со splits.json."""
    expected = cvc.expected_test_subjects(splits_path, [0, 1])
    cvc.validate_fold_subjects(_preds_by_fold([0, 1]), expected)
    swapped = {0: _preds_by_fold([1])[1], 1: _preds_by_fold([0])[0]}
    with pytest.raises(ValueError, match="не совпадают со splits.json"):
        cvc.validate_fold_subjects(swapped, expected)


def test_expected_test_subjects_unknown_fold(splits_path: Path) -> None:
    """Fold, которого нет в splits.json, отвергается."""
    with pytest.raises(ValueError, match="нет outer fold"):
        cvc.expected_test_subjects(splits_path, [0, 7])


def test_macro_is_mean_of_folds_and_pooled_covers_all_subjects() -> None:
    """Macro AUC — среднее AUC fold; pooled охватывает субъектов всех fold."""
    comp = cvc.compare_task_cv("WM", _preds_by_fold([0, 1, 2, 3]), n_bootstrap=200, seed=1)
    assert len(comp.pooled.subjects) == 4 * SUBJECTS_PER_FOLD
    for variant in VARIANTS:
        expected = np.mean([comp.per_fold[k].point[variant] for k in range(4)])
        assert comp.macro.point[variant] == pytest.approx(expected)
        assert comp.macro.replicates[variant].shape == (200,)
    # Менее шумный вариант даёт больший AUC в обоих способах агрегации.
    assert comp.pooled.point["ensemble_full"] > comp.pooled.point["correlation"]
    assert comp.macro.point["ensemble_full"] > comp.macro.point["correlation"]


def test_per_fold_replicates_are_shared_across_variants() -> None:
    """Парность: одинаковые скоры двух вариантов дают одинаковые реплики на fold."""
    fold_preds = _preds_by_fold([0])[0]
    fold_preds["ensemble_full"] = fold_preds["correlation"]
    comp = cvc.compare_task_cv("WM", {0: fold_preds}, n_bootstrap=100, seed=3)
    np.testing.assert_allclose(
        comp.per_fold[0].replicates["correlation"],
        comp.per_fold[0].replicates["ensemble_full"],
    )
    np.testing.assert_allclose(
        comp.macro.replicates["correlation"],
        comp.macro.replicates["ensemble_full"],
    )


def test_run_compare_cv_writes_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    splits_path: Path,
) -> None:
    """Сквозной прогон на синтетике: файлы отчёта, 7 задач + MEAN, фолды 0–2."""
    folds = [0, 1, 2]

    def fake_loader(client, model, task, fold_list, variants, tmp_dir):  # type: ignore[no-untyped-def]
        preds = _preds_by_fold(list(fold_list), task_seed=TASKS.index(task))
        ids = {f"outer{k}_{v}": "run" for k in fold_list for v in variants}
        return preds, ids

    monkeypatch.setattr(cvc, "load_cv_predictions", fake_loader)
    monkeypatch.setattr(cvc, "_mlflow_client", lambda uri: object())
    out_dir = tmp_path / "out"
    code = cvc.run_compare_variants_cv(
        model="skipgcn",
        folds=folds,
        tracking_uri="sqlite:///unused.db",
        splits_path=splits_path,
        out_dir=out_dir,
        n_bootstrap=200,
        log_mlflow=False,
    )
    assert code == 0
    for name in ("auc_pooled.csv", "auc_macro.csv", "auc_per_fold.csv",
                 "pairwise_pooled.csv", "pairwise_macro.csv", "report.md"):
        assert (out_dir / name).is_file(), name
    pooled = pd.read_csv(out_dir / "auc_pooled.csv")
    assert set(pooled["task"]) == {*TASKS, "MEAN"}
    per_fold = pd.read_csv(out_dir / "auc_per_fold.csv")
    assert sorted(per_fold["fold"].unique()) == folds
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert f"субъектов в pooled: {len(folds) * SUBJECTS_PER_FOLD}" in report


def test_run_compare_cv_returns_1_without_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    splits_path: Path,
) -> None:
    """Нет ни одной полной задачи — код 1."""
    monkeypatch.setattr(cvc, "load_cv_predictions", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(cvc, "_mlflow_client", lambda uri: object())
    code = cvc.run_compare_variants_cv(
        model="skipgcn",
        folds=[0, 1],
        tracking_uri="sqlite:///unused.db",
        splits_path=splits_path,
        out_dir=tmp_path / "out",
        n_bootstrap=100,
        log_mlflow=False,
    )
    assert code == 1


def test_stage2_cv_jobs_order_and_names() -> None:
    """Порядок «задача → fold → variant»; outer 0 — прежний experiment, без пересчёта."""
    jobs = build_stage2_cv_jobs("skipgcn", folds=(0, 1), tasks=("WM", "MOTOR"))
    assert len(jobs) == 2 * 2 * len(VARIANTS)
    assert [j.key.split("/")[2] for j in jobs[: 2 * len(VARIANTS)]] == ["WM"] * (2 * len(VARIANTS))
    first = jobs[0]
    assert first.experiment_name == "hcp_stage2_compare_correlation_WM_skipgcn"
    assert "--config-name" not in first.cmd
    fold1 = jobs[len(VARIANTS)]
    assert fold1.experiment_name == "hcp_stage2_cv_correlation_WM_skipgcn_outer1"
    assert "outer_fold=fold1" in fold1.cmd and "config_compare_cv" in fold1.cmd
    assert len({j.experiment_name for j in jobs}) == len(jobs)


def _flatten(node: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in node.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def _hp_equal(actual: Mapping[str, Any], expected: Mapping[str, str]) -> bool:
    for key, want in expected.items():
        if key not in actual:
            return False
        if key == "training.skipgcn.pooling_type":
            equal = actual[key] == want
        else:
            equal = bool(np.isclose(float(actual[key]), float(want)))
        if not equal:
            return False
    return True


def test_frozen_skipgcn_hyperparameters() -> None:
    """HP зафиксированы: 27 ячеек — дефолтные, correlation/EMOTION — подобранные на outer 0."""
    conf = Path(__file__).resolve().parents[1] / "conf" / "best_task"
    for variant in VARIANTS:
        for task in TASKS:
            cfg = OmegaConf.load(conf / f"skipgcn_{variant}_{task}.yaml")
            actual = _flatten(OmegaConf.to_container(cfg))  # type: ignore[arg-type]
            expected = (
                SKIPGCN_CORR_AVG_BEST_HP
                if (variant, task) == ("correlation", "EMOTION")
                else SKIPGCN_DEFAULT_HP
            )
            assert _hp_equal(actual, expected), (variant, task)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"folds": (4,)},
        {"folds": ()},
        {"tasks": ("NOPE",)},
        {"model": "nope"},
    ],
)
def test_stage2_cv_jobs_validation(kwargs: dict) -> None:
    """Некорректные аргументы отвергаются сразу."""
    with pytest.raises(ValueError):
        build_stage2_cv_jobs(**kwargs)


@pytest.mark.parametrize("fold", OUTER_FOLDS)
def test_outer_fold_group_selects_fold(fold: int) -> None:
    """Группа outer_fold=foldK задаёт data.outer_fold, имя experiment и зафиксированные HP."""
    from hydra import compose, initialize_config_dir

    conf = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(config_dir=str(conf), version_base=None):
        cfg = compose(
            config_name="config_compare_cv",
            overrides=[f"outer_fold=fold{fold}", "variant=ensemble_full", "task=WM"],
        )
        assert int(cfg.data.outer_fold) == fold
        assert cfg.mlflow.experiment_name == (
            f"hcp_stage2_cv_ensemble_full_WM_skipgcn_outer{fold}"
        )
        assert float(cfg.training.lr) == 0.001


def test_outer_fold_group_is_mandatory() -> None:
    """Без outer_fold конфиг не собирается (иначе crossval обошёл бы все fold в одном run)."""
    from hydra import compose, initialize_config_dir
    from hydra.errors import ConfigCompositionException

    conf = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(config_dir=str(conf), version_base=None):
        with pytest.raises(ConfigCompositionException):
            compose(config_name="config_compare_cv", overrides=["variant=ensemble_full", "task=WM"])


@pytest.mark.parametrize("fold", OUTER_FOLDS)
def test_frozen_hp_do_not_depend_on_fold(fold: int) -> None:
    """correlation/EMOTION получает подобранные на outer 0 HP на любом fold."""
    from hydra import compose, initialize_config_dir

    conf = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(config_dir=str(conf), version_base=None):
        cfg = compose(
            config_name="config_compare_cv",
            overrides=[f"outer_fold=fold{fold}", "variant=correlation", "task=EMOTION"],
        )
    assert float(cfg.training.lr) == pytest.approx(0.00158)
    assert int(cfg.data.batch_size) == 16
