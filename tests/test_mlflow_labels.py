"""Тесты различимых меток MLflow."""

from __future__ import annotations

from omegaconf import OmegaConf

from mlflow_utils import _resolve_run_name, run_tags


def test_explicit_run_name_kept() -> None:
    """Явное имя run сохраняется и содержит модель, вариант и задачу."""
    cfg = OmegaConf.create(
        {
            "training": {"model_name": "skipgcn"},
            "data": {"variant": "correlation", "task": "WM"},
            "mlflow": {
                "experiment_name": "hcp_smoke_defaults_correlation_WM_skipgcn",
                "run_name": "smoke_skipgcn_correlation_WM",
            },
        }
    )
    tags = run_tags(cfg)
    assert tags["model"] == "skipgcn"
    assert tags["variant"] == "correlation"
    assert tags["task"] == "WM"
    assert tags["experiment"] == "hcp_smoke_defaults_correlation_WM_skipgcn"
    assert _resolve_run_name(cfg) == "smoke_skipgcn_correlation_WM"


def test_missing_run_name_uses_model_variant_task() -> None:
    """Пустое имя run собирается из модели, варианта и задачи."""
    cfg = OmegaConf.create(
        {
            "training": {"model_name": "cfmsr"},
            "data": {"variant": "ensemble_full", "task": "EMOTION"},
            "debug_check": "lr",
            "mlflow": {"run_name": None},
        }
    )
    assert _resolve_run_name(cfg) == "debug_lr_cfmsr_ensemble_full_EMOTION"
    assert run_tags(cfg)["debug_check"] == "lr"
