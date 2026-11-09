"""Интеграция MLflow для обучения и кросс-валидации."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def get_mlflow_cfg(cfg: DictConfig) -> Optional[DictConfig]:
    """
    Get mlflow cfg.

    Parameters
    ----------
    cfg : DictConfig
        Полная Hydra-конфигурация.

    Returns
    -------
    Optional[DictConfig]
        Ветка ``mlflow`` или ``None``.
    """
    if "mlflow" not in cfg:
        return None
    return cfg.mlflow


def _is_mlflow_enabled(cfg: DictConfig) -> bool:
    mlflow_cfg = get_mlflow_cfg(cfg)
    if mlflow_cfg is None:
        return False
    return bool(mlflow_cfg.get("enabled", False))


def _project_root() -> Path:
    try:
        from hydra.utils import get_original_cwd

        return Path(get_original_cwd()).resolve()
    except Exception:
        return Path.cwd().resolve()


def resolve_sqlite_uri(uri: str) -> str:
    """
    Resolve sqlite uri.

    Parameters
    ----------
    uri : str
        Исходный tracking URI.

    Returns
    -------
    str
        URI с абсолютным путём к файлу БД.
    """
    if not uri.startswith("sqlite:///"):
        return uri
    db_path = uri[len("sqlite:///") :]
    path = Path(db_path).expanduser()
    if not path.is_absolute():
        path = _project_root() / path
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def _cfg_label(cfg: DictConfig, *keys: str) -> str:
    node: Any = cfg
    for key in keys:
        if node is None or key not in node:
            return ""
        node = node[key]
    if node is None:
        return ""
    text = str(node).strip()
    if text.lower() in {"", "none", "null"}:
        return ""
    return text


def run_tags(cfg: DictConfig) -> Dict[str, str]:
    """
    Run tags.

    Метки, по которым run отличается от соседних: модель, вариант, задача.

    Parameters
    ----------
    cfg : DictConfig
        Полная Hydra-конфигурация.

    Returns
    -------
    Dict[str, str]
        Непустые теги.
    """
    tags = {
        "model": _cfg_label(cfg, "training", "model_name"),
        "variant": _cfg_label(cfg, "data", "variant"),
        "task": _cfg_label(cfg, "data", "task"),
        "debug_check": _cfg_label(cfg, "debug_check"),
    }
    if _cfg_label(cfg, "data", "shuffle_labels").lower() in {"true", "1", "yes"}:
        tags["shuffle_labels"] = "true"
        tags["shuffle_labels_scope"] = _cfg_label(cfg, "data", "shuffle_labels_scope") or "all"
        tags["shuffle_labels_seed"] = _cfg_label(cfg, "data", "shuffle_labels_seed") or "0"
    mlflow_cfg = get_mlflow_cfg(cfg)
    if mlflow_cfg is not None:
        tags["experiment"] = _cfg_label(mlflow_cfg, "experiment_name")
    if "hpo" in cfg:
        tags["hpo_study"] = _cfg_label(cfg, "hpo", "study_name")
    return {key: value for key, value in tags.items() if value}


def _default_run_name(cfg: DictConfig) -> str:
    tags = run_tags(cfg)
    if tags.get("debug_check"):
        parts = [
            "debug",
            tags["debug_check"],
            tags.get("model", ""),
            tags.get("variant", ""),
            tags.get("task", ""),
        ]
    else:
        parts = [tags.get("model", ""), tags.get("variant", ""), tags.get("task", "")]
    name = "_".join(part for part in parts if part)
    return name or "train"


def _append_trial(name: str) -> str:
    """
    Append trial index.

    В Optuna multirun к имени добавляется номер trial, иначе все trial
    получают одну и ту же метку.
    """
    try:
        from hydra.core.hydra_config import HydraConfig
        from hydra.types import RunMode

        hydra_cfg = HydraConfig.get()
        if hydra_cfg.mode == RunMode.MULTIRUN:
            return f"trial_{hydra_cfg.job.num}_{name}"
    except Exception:
        return name
    return name


def _resolve_run_name(cfg: DictConfig) -> str:
    mlflow_cfg = get_mlflow_cfg(cfg)
    if mlflow_cfg is not None:
        explicit = mlflow_cfg.get("run_name", None)
        if explicit is not None and str(explicit).strip() not in {"", "none", "null"}:
            return _append_trial(str(explicit).strip())
    return _append_trial(_default_run_name(cfg))


@contextmanager
def mlflow_run_context(
    cfg: DictConfig,
    run_name: Optional[str] = None,
    nested: bool = False,
) -> Iterator[Optional[Any]]:
    """
    MLflow run context.

    Parameters
    ----------
    cfg : DictConfig
        Полная Hydra-конфигурация.
    run_name : Optional[str], optional
        Явное имя run.
    nested : bool, optional
        Вложенный run (для fold).

    Yields
    ------
    Optional[Any]
        Активный MLflow run или ``None``.
    """
    if not _is_mlflow_enabled(cfg):
        yield None
        return

    import mlflow

    mlflow_cfg = get_mlflow_cfg(cfg)
    assert mlflow_cfg is not None

    tracking_uri = resolve_sqlite_uri(
        str(mlflow_cfg.get("tracking_uri", "sqlite:///artifacts/mlflow.db"))
    )
    experiment_name = str(mlflow_cfg.get("experiment_name", "hcp_gnn_compare"))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    resolved_name = (
        _append_trial(run_name) if run_name is not None else _resolve_run_name(cfg)
    )

    with mlflow.start_run(run_name=resolved_name, nested=nested) as run:
        mlflow.set_tags(run_tags(cfg))
        log.info(
            "MLflow run started: experiment=%s, run_id=%s, nested=%s",
            experiment_name,
            run.info.run_id,
            nested,
        )
        yield run


def _flatten_branch(branch: Any, prefix: str, flat: Dict[str, Any]) -> None:
    if not isinstance(branch, dict):
        flat[prefix] = branch
        return
    for key, value in branch.items():
        if key == "metrics":
            continue
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            _flatten_branch(value, full, flat)
        else:
            flat[full] = value


def log_hparams(cfg: DictConfig) -> None:
    """
    Log hparams.

    Parameters
    ----------
    cfg : DictConfig
        Полная Hydra-конфигурация.
    """
    if not _is_mlflow_enabled(cfg):
        return

    import mlflow

    flat: Dict[str, Any] = {}
    for branch_name in ("data", "training", "cv"):
        if branch_name not in cfg:
            continue
        branch = OmegaConf.to_container(cfg[branch_name], resolve=True)
        if not isinstance(branch, dict):
            continue
        _flatten_branch(branch, branch_name, flat)

    safe = {}
    for key, value in flat.items():
        s = _mlflow_safe_value(value)
        if isinstance(s, str) and len(s) > 250:
            s = s[:247] + "..."
        safe[key] = s
    mlflow.log_params(safe)


def log_split_metrics(
    split_name: str,
    metrics: Dict[str, Any],
    metric_names: List[str],
    step: Optional[int] = None,
) -> None:
    """
    Log split metrics.

    Parameters
    ----------
    split_name : str
        Имя split.
    metrics : Dict[str, Any]
        Словарь метрик.
    metric_names : List[str]
        Список имён метрик.
    step : Optional[int], optional
        Номер эпохи.
    """
    import mlflow

    if mlflow.active_run() is None:
        return

    prefix = f"{split_name}_"
    kwargs: Dict[str, Any] = {}
    if step is not None:
        kwargs["step"] = int(step)

    if "loss" in metrics:
        try:
            loss = float(metrics["loss"])
            if not pd.isna(loss):
                mlflow.log_metric(f"{prefix}loss", loss, **kwargs)
        except Exception:
            pass

    for name in metric_names:
        if name not in metrics:
            continue
        try:
            value = float(metrics[name])
        except Exception:
            continue
        if pd.isna(value):
            continue
        mlflow.log_metric(f"{prefix}{name}", value, **kwargs)


def log_overfit_gap_metrics(
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    metric_names: List[str],
    step: int,
) -> None:
    """
    Log overfit gap metrics.

    Parameters
    ----------
    train_metrics : Dict[str, Any]
        Метрики train.
    val_metrics : Dict[str, Any]
        Метрики val.
    metric_names : List[str]
        Список имён метрик.
    step : int
        Номер эпохи.
    """
    import mlflow

    if mlflow.active_run() is None:
        return

    for name in ["loss", *metric_names]:
        if name not in train_metrics or name not in val_metrics:
            continue
        try:
            train_value = float(train_metrics[name])
            val_value = float(val_metrics[name])
        except Exception:
            continue
        if pd.isna(train_value) or pd.isna(val_value):
            continue
        gap = val_value - train_value if name == "loss" else train_value - val_value
        mlflow.log_metric(f"gap_{name}", float(gap), step=int(step))


def log_artifact_path(cfg: DictConfig, path: Optional[str]) -> None:
    """
    Log artifact path.

    Parameters
    ----------
    cfg : DictConfig
        Полная Hydra-конфигурация.
    path : Optional[str]
        Путь к файлу.
    """
    if not _is_mlflow_enabled(cfg) or path is None:
        return
    mlflow_cfg = get_mlflow_cfg(cfg)
    if mlflow_cfg is None or not bool(mlflow_cfg.get("log_artifacts", True)):
        return
    file_path = Path(path)
    if not file_path.is_file():
        return
    import mlflow

    if mlflow.active_run() is None:
        return
    mlflow.log_artifact(str(file_path))


def _mlflow_safe_value(value: Any) -> Any:
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
