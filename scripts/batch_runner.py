#!/usr/bin/env python3
"""
Batch runner with resume.

Последовательный запуск экспериментов с возобновлением после выключения:
- FINISHED — пропускаются;
- FAILED / RUNNING (обрыв) — удаляются из MLflow;
- старт с первого незавершённого после непрерывного префикса FINISHED.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("batch_runner")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = f"sqlite:///{ROOT / 'artifacts' / 'mlflow.db'}"
PROGRESS_DIR = ROOT / "artifacts" / "batch_progress"
HPO_TARGETS_PATH = ROOT / "conf" / "hpo_targets" / "default.yaml"
DEBUG_CFMSR_CHECKS = (
    "baseline",
    "lr",
    "warmup",
    "edge_scaler",
    "node_features",
)
DEBUG_CFMSR_VARIANT = "ensemble_correlation"
DEBUG_CFMSR_TASK = "EMOTION"

VARIANTS = (
    "correlation",
    "ensemble_correlation",
    "ensemble_univariate",
    "ensemble_full",
)
TASKS = (
    "EMOTION",
    "GAMBLING",
    "MOTOR",
    "LANGUAGE",
    "SOCIAL",
    "RELATIONAL",
    "WM",
)
MODELS = ("skipgcn", "cfmsr")
# Outer fold архива для stage2_cv.
OUTER_FOLDS = (0, 1, 2, 3)
ENSEMBLE_VARIANTS = (
    "ensemble_correlation",
    "ensemble_univariate",
    "ensemble_full",
)
# Ячейки с AUC≈1.0 в stage-2 отчёте — цель проверки утечки.
LEAK_CHECK_VARIANTS = ("ensemble_univariate", "ensemble_full")
LEAK_CHECK_TASKS = ("EMOTION", "MOTOR", "SOCIAL")
LEAK_CHECK_SCOPES = ("all", "train")


@dataclass(frozen=True)
class Job:
    """
    Job.

    Attributes
    ----------
    key : str
        Уникальный ключ эксперимента.
    experiment_name : str
        Имя MLflow experiment.
    cmd : List[str]
        Команда запуска (без интерпретатора).
    optuna_study : Optional[str]
        Имя Optuna study (только stage1); ``None`` если не HPO.
    match_params : Optional[Dict[str, str]]
        Для confirm: FINISHED parent должен совпасть по этим params.
    """

    key: str
    experiment_name: str
    cmd: List[str]
    optuna_study: Optional[str] = None
    match_params: Optional[Dict[str, str]] = None


# Дефолты smoke / контроль confirm A.
SKIPGCN_DEFAULT_HP: Dict[str, str] = {
    "training.lr": "0.001",
    "training.weight_decay": "0.0001",
    "training.skipgcn.hidden_channels": "32",
    "training.skipgcn.dropout": "0.2",
    "training.skipgcn.pooling_type": "global_max_pool",
    "data.drop_percentile": "50",
    "data.batch_size": "8",
}

# Средние best correlation (confirm B).
SKIPGCN_CORR_AVG_BEST_HP: Dict[str, str] = {
    "training.lr": "0.00158",
    "training.weight_decay": "3.7e-05",
    "training.skipgcn.hidden_channels": "32",
    "training.skipgcn.dropout": "0.25",
    "training.skipgcn.pooling_type": "global_max_pool",
    "data.drop_percentile": "50.0",
    "data.batch_size": "16",
}

# Per-task лучший trial отсева (confirm C). RELATIONAL ensemble_correlation пропущен.
SKIPGCN_PER_TASK_BEST: Dict[tuple, Dict[str, str]] = {
    ("correlation", "EMOTION"): {
        "training.lr": "0.0019247055",
        "training.weight_decay": "4.0027109e-05",
        "training.skipgcn.hidden_channels": "24",
        "training.skipgcn.dropout": "0.2036",
        "training.skipgcn.pooling_type": "global_mean_pool",
        "data.drop_percentile": "90",
        "data.batch_size": "16",
    },
    ("correlation", "MOTOR"): {
        "training.lr": "0.002091928",
        "training.weight_decay": "3.4191308e-05",
        "training.skipgcn.hidden_channels": "48",
        "training.skipgcn.dropout": "0.2842",
        "training.skipgcn.pooling_type": "global_max_pool",
        "data.drop_percentile": "50",
        "data.batch_size": "16",
    },
    ("ensemble_correlation", "GAMBLING"): {
        "training.lr": "0.0029190419",
        "training.weight_decay": "4.2095087e-05",
        "training.skipgcn.hidden_channels": "32",
        "training.skipgcn.dropout": "0.2961",
        "training.skipgcn.pooling_type": "global_mean_pool",
        "data.drop_percentile": "50",
        "data.batch_size": "16",
    },
    ("ensemble_correlation", "SOCIAL"): {
        "training.lr": "0.00063492819",
        "training.weight_decay": "3.8929554e-05",
        "training.skipgcn.hidden_channels": "24",
        "training.skipgcn.dropout": "0.0531",
        "training.skipgcn.pooling_type": "global_mean_pool",
        "data.drop_percentile": "90",
        "data.batch_size": "16",
    },
    ("ensemble_correlation", "WM"): {
        "training.lr": "0.00057486995",
        "training.weight_decay": "0.0093512549",
        "training.skipgcn.hidden_channels": "32",
        "training.skipgcn.dropout": "0.3554",
        "training.skipgcn.pooling_type": "global_max_pool",
        "data.drop_percentile": "25",
        "data.batch_size": "16",
    },
    ("ensemble_univariate", "GAMBLING"): {
        "training.lr": "0.0016155001",
        "training.weight_decay": "0.0038211294",
        "training.skipgcn.hidden_channels": "16",
        "training.skipgcn.dropout": "0.2019",
        "training.skipgcn.pooling_type": "global_max_pool",
        "data.drop_percentile": "75",
        "data.batch_size": "8",
    },
    ("ensemble_univariate", "RELATIONAL"): {
        "training.lr": "0.0016155001",
        "training.weight_decay": "0.0038211294",
        "training.skipgcn.hidden_channels": "16",
        "training.skipgcn.dropout": "0.2019",
        "training.skipgcn.pooling_type": "global_max_pool",
        "data.drop_percentile": "75",
        "data.batch_size": "8",
    },
    ("ensemble_full", "GAMBLING"): {
        "training.lr": "0.00082948311",
        "training.weight_decay": "0.00013824387",
        "training.skipgcn.hidden_channels": "48",
        "training.skipgcn.dropout": "0.4905",
        "training.skipgcn.pooling_type": "global_max_pool",
        "data.drop_percentile": "90",
        "data.batch_size": "4",
    },
    ("ensemble_full", "RELATIONAL"): {
        "training.lr": "0.0016012775",
        "training.weight_decay": "0.0013332308",
        "training.skipgcn.hidden_channels": "32",
        "training.skipgcn.dropout": "0.1876",
        "training.skipgcn.pooling_type": "global_max_pool",
        "data.drop_percentile": "0",
        "data.batch_size": "4",
    },
}

ENSEMBLE_CONFIRM_TARGETS: tuple = (
    ("ensemble_correlation", "GAMBLING"),
    ("ensemble_correlation", "SOCIAL"),
    ("ensemble_correlation", "RELATIONAL"),
    ("ensemble_correlation", "WM"),
    ("ensemble_univariate", "GAMBLING"),
    ("ensemble_univariate", "RELATIONAL"),
    ("ensemble_full", "GAMBLING"),
    ("ensemble_full", "RELATIONAL"),
)

CORR_AVG_BEST_TASKS: tuple = ("EMOTION", "MOTOR", "RELATIONAL", "WM")
# Дефолты на correlation для сравнения с avg best (GAMBLING/SOCIAL уже есть).
CORR_DEFAULT_CONTROL_TASKS: tuple = ("EMOTION", "MOTOR", "RELATIONAL", "WM")


def _resolve_python() -> str:
    geometric = Path("/home/vadim/anaconda3/envs/geometric/bin/python")
    if geometric.is_file():
        return str(geometric)
    return sys.executable


def _mlflow_client(tracking_uri: str):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def _list_runs(client, experiment_name: str) -> List[object]:
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return []
    return client.search_runs(
        experiment_ids=[exp.experiment_id],
        run_view_type=1,  # ACTIVE_ONLY
        max_results=5000,
    )


def _run_status(run: object) -> str:
    """
    Run status string.

    Parameters
    ----------
    run : object
        MLflow run.

    Returns
    -------
    str
        Статус в верхнем регистре (FINISHED / FAILED / ...).
    """
    from mlflow.entities import RunStatus

    raw = run.info.status
    if isinstance(raw, str):
        return raw.upper()
    try:
        return str(RunStatus.to_string(raw)).upper()
    except Exception:
        return str(raw).upper()


def delete_failed_and_running(client, experiment_name: str) -> int:
    """
    Delete failed and running.

    Удаляет runs со статусом FAILED, RUNNING и KILLED (прерванные).

    Parameters
    ----------
    client : MlflowClient
        Клиент MLflow.
    experiment_name : str
        Имя эксперимента.

    Returns
    -------
    int
        Число удалённых runs.
    """
    deleted = 0
    for run in _list_runs(client, experiment_name):
        status = _run_status(run)
        if status in {"FAILED", "RUNNING", "KILLED"}:
            log.info(
                "Удаляю run %s status=%s experiment=%s",
                run.info.run_id,
                status,
                experiment_name,
            )
            client.delete_run(run.info.run_id)
            deleted += 1
    return deleted


def _params_match(run_params: Dict[str, str], expected: Dict[str, str]) -> bool:
    """
    Params match.

    Числовые params сравниваются с относительной погрешностью.
    """
    for key, want in expected.items():
        got = run_params.get(key)
        if got is None:
            return False
        try:
            g = float(got)
            w = float(want)
        except ValueError:
            if str(got) != str(want):
                return False
            continue
        scale = max(abs(w), abs(g), 1e-12)
        if abs(g - w) / scale > 1e-5 and abs(g - w) > 1e-8:
            return False
    return True


def has_finished_matching_parent(
    client,
    experiment_name: str,
    match_params: Optional[Dict[str, str]] = None,
) -> bool:
    """
    Has finished matching parent.

    Parameters
    ----------
    client : MlflowClient
        Клиент MLflow.
    experiment_name : str
        Имя эксперимента.
    match_params : Optional[Dict[str, str]], optional
        Если заданы — parent должен совпасть по params.

    Returns
    -------
    bool
        True, если есть подходящий FINISHED parent.
    """
    runs = _list_runs(client, experiment_name)
    for run in runs:
        if _run_status(run) != "FINISHED":
            continue
        if run.data.tags.get("mlflow.parentRunId") is not None:
            continue
        if match_params is None:
            return True
        if _params_match(dict(run.data.params), match_params):
            return True
    return False


def has_finished_parent_run(client, experiment_name: str) -> bool:
    """
    Has finished parent run.

    Эксперимент считается завершённым, если есть хотя бы один
    корневой (не nested) run со статусом FINISHED.

    Parameters
    ----------
    client : MlflowClient
        Клиент MLflow.
    experiment_name : str
        Имя эксперимента.

    Returns
    -------
    bool
        True, если есть завершённый родительский run.
    """
    runs = _list_runs(client, experiment_name)
    for run in runs:
        if _run_status(run) != "FINISHED":
            continue
        # Nested fold-runs имеют parent; предпочитаем корневой run.
        parent = run.data.tags.get("mlflow.parentRunId")
        if parent is None:
            return True
    # Fallback: любой FINISHED (если логировали только nested).
    return any(_run_status(run) == "FINISHED" for run in runs)


def optuna_complete_count(storage: str, study_name: str) -> int:
    """
    Optuna complete count.

    Parameters
    ----------
    storage : str
        URI хранилища Optuna.
    study_name : str
        Имя study.

    Returns
    -------
    int
        Число COMPLETE trials; 0, если study нет.
    """
    try:
        import optuna
    except ImportError:
        log.warning("optuna не установлен — статус study не проверяю")
        return 0

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception:
        return 0
    return sum(1 for t in study.trials if t.state.name == "COMPLETE")


def optuna_remaining_trials(
    storage: str,
    study_name: str,
    n_trials: int,
) -> int:
    """
    Optuna remaining trials.

    Hydra Optuna sweeper трактует ``n_trials`` как число *новых* trials
    в текущем запуске. Поэтому при resume передаём остаток, а не цель.

    Parameters
    ----------
    storage : str
        URI хранилища Optuna.
    study_name : str
        Имя study.
    n_trials : int
        Целевое суммарное число COMPLETE trials.

    Returns
    -------
    int
        Сколько ещё trials нужно запустить (≥ 0).
    """
    n_complete = optuna_complete_count(storage, study_name)
    remaining = max(0, int(n_trials) - n_complete)
    log.info(
        "Optuna study=%s complete=%d / %d remaining=%d",
        study_name,
        n_complete,
        n_trials,
        remaining,
    )
    return remaining


def cleanup_optuna_study(
    storage: str,
    study_name: str,
    *,
    delete_if_empty: bool = True,
) -> int:
    """
    Cleanup Optuna study.

    RUNNING trials после обрыва помечаются FAIL. Study без COMPLETE
    удаляется, чтобы следующий запуск начинал с чистого имени.

    Parameters
    ----------
    storage : str
        URI хранилища Optuna.
    study_name : str
        Имя study.
    delete_if_empty : bool, optional
        Удалить study, если нет COMPLETE trials.

    Returns
    -------
    int
        Число trials, переведённых в FAIL.
    """
    try:
        import optuna
        from optuna.trial import TrialState
    except ImportError:
        return 0

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception:
        return 0

    n_failed = 0
    for trial in study.get_trials(deepcopy=False):
        if trial.state != TrialState.RUNNING:
            continue
        try:
            study._storage.set_trial_state(trial._trial_id, TrialState.FAIL)
            n_failed += 1
            log.info(
                "Optuna study=%s trial=%d RUNNING → FAIL",
                study_name,
                trial.number,
            )
        except Exception as exc:
            log.warning(
                "Не удалось закрыть RUNNING trial %d study=%s: %s",
                trial.number,
                study_name,
                exc,
            )

    if delete_if_empty:
        n_complete = sum(
            1 for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE
        )
        if n_complete == 0:
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
                log.info("Optuna study=%s удалён (0 COMPLETE)", study_name)
            except Exception as exc:
                log.warning("Не удалось удалить study=%s: %s", study_name, exc)

    return n_failed


def optuna_study_complete(
    storage: str,
    study_name: str,
    n_trials: int,
) -> bool:
    """
    Optuna study complete.

    Parameters
    ----------
    storage : str
        URI хранилища Optuna.
    study_name : str
        Имя study.
    n_trials : int
        Целевое число trials.

    Returns
    -------
    bool
        True, если число COMPLETE trials >= ``n_trials``.
    """
    n_complete = optuna_complete_count(storage, study_name)
    log.info(
        "Optuna study=%s complete_trials=%d / %d",
        study_name,
        n_complete,
        n_trials,
    )
    return n_complete >= int(n_trials)


def job_with_n_trials(job: Job, n_trials: int) -> Job:
    """
    Job with n trials.

    Переписывает ``hpo.n_trials=`` в команде на остаток trials.

    Parameters
    ----------
    job : Job
        Исходный job.
    n_trials : int
        Число trials для *этого* запуска.

    Returns
    -------
    Job
        Копия job с обновлённой командой.
    """
    cmd: List[str] = []
    replaced = False
    for arg in job.cmd:
        if arg.startswith("hpo.n_trials="):
            cmd.append(f"hpo.n_trials={int(n_trials)}")
            replaced = True
        else:
            cmd.append(arg)
    if not replaced:
        cmd.append(f"hpo.n_trials={int(n_trials)}")
    return Job(
        key=job.key,
        experiment_name=job.experiment_name,
        cmd=cmd,
        optuna_study=job.optuna_study,
        match_params=job.match_params,
    )


def _hp_to_overrides(hp: Dict[str, str]) -> List[str]:
    return [f"{key}={value}" for key, value in hp.items()]


def _confirm_job(
    phase: str,
    variant: str,
    task: str,
    *,
    hp: Dict[str, str],
    extra_overrides: Optional[Sequence[str]] = None,
) -> Job:
    """
    Confirm job.

    Parameters
    ----------
    phase : str
        Метка фазы (A / B / C).
    variant : str
        Вариант графа.
    task : str
        Задача.
    hp : Dict[str, str]
        Гиперпараметры для CLI и match.
    extra_overrides : Optional[Sequence[str]], optional
        Доп. Hydra overrides (например ``~best``).

    Returns
    -------
    Job
        Confirm job.
    """
    cmd = [
        "main.py",
        "--config-name",
        "config_hpo_confirm",
        "model=skipgcn",
        f"variant={variant}",
        f"task={task}",
    ]
    if extra_overrides:
        cmd.extend(extra_overrides)
    cmd.extend(_hp_to_overrides(hp))
    return Job(
        key=f"confirm_{phase}/{variant}/{task}",
        experiment_name=f"hcp_hpo_confirm_skipgcn_{variant}_{task}",
        cmd=cmd,
        match_params=dict(hp),
    )


def build_confirm_a_jobs() -> List[Job]:
    """
    Build confirm A jobs.

    Дефолты smoke: 8 ensemble-targets + 4 оставшихся correlation
    (для сравнения с avg best в B). На correlation нужен ``~best``.

    Returns
    -------
    List[Job]
        Контрольные confirm-прогоны.
    """
    jobs: List[Job] = []
    for task in CORR_DEFAULT_CONTROL_TASKS:
        jobs.append(
            _confirm_job(
                "A",
                "correlation",
                task,
                hp=SKIPGCN_DEFAULT_HP,
                extra_overrides=("~best", "~best_task"),
            )
        )
    for variant, task in ENSEMBLE_CONFIRM_TARGETS:
        jobs.append(
            _confirm_job(
                "A",
                variant,
                task,
                hp=SKIPGCN_DEFAULT_HP,
                extra_overrides=("~best_task",),
            )
        )
    return jobs


def build_confirm_b_jobs() -> List[Job]:
    """
    Build confirm B jobs.

    Средние best на оставшихся correlation-задачах. HP передаются через CLI
    и используются для match в MLflow.

    Returns
    -------
    List[Job]
        Confirm со средними best.
    """
    return [
        _confirm_job(
            "B",
            "correlation",
            task,
            hp=SKIPGCN_CORR_AVG_BEST_HP,
            extra_overrides=("~best_task",),
        )
        for task in CORR_AVG_BEST_TASKS
    ]


def build_confirm_c_jobs() -> List[Job]:
    """
    Build confirm C jobs.

    Per-task лучший trial отсева. На correlation нужен ``~best``, чтобы
    средние best не перебили trial.

    Returns
    -------
    List[Job]
        Confirm с per-task HP.
    """
    jobs: List[Job] = []
    for (variant, task), hp in SKIPGCN_PER_TASK_BEST.items():
        extra: Tuple[str, ...]
        if variant == "correlation":
            extra = ("~best", "~best_task")
        else:
            extra = ("~best_task",)
        jobs.append(
            _confirm_job("C", variant, task, hp=hp, extra_overrides=extra)
        )
    return jobs


def load_hpo_targets(model_name: str) -> List[Dict[str, Any]]:
    """
    Load HPO targets.

    Parameters
    ----------
    model_name : str
        ``skipgcn`` или ``cfmsr``.

    Returns
    -------
    List[Dict[str, Any]]
        Пары variant×task со smoke ROC-AUC < 0.99.
    """
    from omegaconf import OmegaConf

    if not HPO_TARGETS_PATH.is_file():
        raise FileNotFoundError(f"Нет файла targets: {HPO_TARGETS_PATH}")
    cfg = OmegaConf.load(HPO_TARGETS_PATH)
    if model_name not in cfg:
        raise KeyError(f"В {HPO_TARGETS_PATH.name} нет ключа {model_name}")
    items = OmegaConf.to_container(cfg[model_name], resolve=True)
    if not isinstance(items, list) or not items:
        raise ValueError(f"Пустой список targets для {model_name}")
    return list(items)


def resolve_variant_filter(
    raw: Optional[Sequence[str]],
) -> Optional[set[str]]:
    """
    Resolve variant filter.

    Parameters
    ----------
    raw : Optional[Sequence[str]]
        Значения ``--variant``; ``ensemble`` раскрывается в три ensemble_*.

    Returns
    -------
    Optional[set]
        Множество variant или ``None``, если фильтра нет.
    """
    if not raw:
        return None
    resolved: set[str] = set()
    known = set(VARIANTS) | {"ensemble"}
    for item in raw:
        name = str(item).strip()
        if name not in known:
            raise SystemExit(
                f"Неизвестный variant={name}. "
                f"Допустимы: {', '.join(VARIANTS)} или ensemble"
            )
        if name == "ensemble":
            resolved.update(ENSEMBLE_VARIANTS)
        else:
            resolved.add(name)
    return resolved


def job_variant(job: Job) -> Optional[str]:
    """
    Job variant.

    Parameters
    ----------
    job : Job
        Job batch-матрицы.

    Returns
    -------
    Optional[str]
        variant из ключа или ``None``.
    """
    parts = job.key.split("/")
    if parts[0] == "smoke" and len(parts) >= 2:
        return parts[1]
    if parts[0] == "stage0" and len(parts) >= 3 and parts[1] == "skipgcn":
        return parts[2]
    if parts[0].startswith("confirm_") and len(parts) >= 2:
        return parts[1]
    if parts[0] in {"stage1", "stage2", "stage2_cv"} and len(parts) >= 2:
        return parts[1]
    if parts[0] == "leak_check" and len(parts) >= 3:
        return parts[2]
    return None


def filter_jobs_by_variant(
    jobs: Sequence[Job],
    variants: Optional[set[str]],
) -> List[Job]:
    """
    Filter jobs by variant.

    Parameters
    ----------
    jobs : Sequence[Job]
        Полный список jobs.
    variants : Optional[set]
        Оставляемые variant.

    Returns
    -------
    List[Job]
        Отфильтрованные jobs.
    """
    if variants is None:
        return list(jobs)
    return [job for job in jobs if job_variant(job) in variants]


def build_smoke_jobs() -> List[Job]:
    """
    Build smoke jobs.

    Returns
    -------
    List[Job]
        Матрица 2×4×7 для smoke.
    """
    jobs: List[Job] = []
    for variant in VARIANTS:
        for task in TASKS:
            for model in MODELS:
                key = f"smoke/{variant}/{task}/{model}"
                exp = f"hcp_smoke_defaults_{variant}_{task}_{model}"
                cmd = [
                    "main_smoke.py",
                    f"model={model}",
                    f"variant={variant}",
                    f"task={task}",
                ]
                jobs.append(Job(key=key, experiment_name=exp, cmd=cmd))
    return jobs


def build_stage0_skipgcn_jobs(n_trials: int) -> List[Job]:
    """
    Build stage0 SkipGCN jobs.

    Отсев (screen) только на комбинациях из ``hpo_targets``:
    outer fold 0, один inner, patience=10.

    Parameters
    ----------
    n_trials : int
        Число Optuna trials на study.

    Returns
    -------
    List[Job]
        Jobs SkipGCN по targets.
    """
    jobs: List[Job] = []
    for item in load_hpo_targets("skipgcn"):
        variant = str(item["variant"])
        task = str(item["task"])
        key = f"stage0/skipgcn/{variant}/{task}"
        exp = f"hcp_hpo_screen_skipgcn_{variant}_{task}"
        study = f"skipgcn_screen_{variant}_{task}"
        cmd = [
            "hpo_skipgcn.py",
            "-m",
            f"variant={variant}",
            f"task={task}",
            f"hpo.n_trials={n_trials}",
        ]
        jobs.append(
            Job(
                key=key,
                experiment_name=exp,
                cmd=cmd,
                optuna_study=study,
            )
        )
    return jobs


def build_stage0_cfmsr_jobs() -> List[Job]:
    """
    Build stage0 CF-MSR jobs.

    Пять однофакторных проверок коллапса на ensemble_correlation / EMOTION.
    Подбор CF-MSR на этом варианте не запускается.

    Returns
    -------
    List[Job]
        Debug jobs CF-MSR.
    """
    jobs: List[Job] = []
    for check in DEBUG_CFMSR_CHECKS:
        key = f"stage0/cfmsr/debug/{check}"
        exp = (
            f"hcp_debug_cfmsr_{check}"
            f"_{DEBUG_CFMSR_VARIANT}_{DEBUG_CFMSR_TASK}"
        )
        cmd = [
            "main.py",
            "--config-name",
            "config_debug_cfmsr",
            f"experiment=debug_cfmsr/{check}",
        ]
        jobs.append(Job(key=key, experiment_name=exp, cmd=cmd))
    return jobs


def build_stage1_jobs(n_trials: int) -> List[Job]:
    """
    Build stage1 jobs.

    Parameters
    ----------
    n_trials : int
        Число Optuna trials.

    Returns
    -------
    List[Job]
        4×7 HPO jobs.
    """
    jobs: List[Job] = []
    for variant in VARIANTS:
        for task in TASKS:
            key = f"stage1/{variant}/{task}"
            exp = f"hcp_hpo_screen_skipgcn_{variant}_{task}"
            study = f"skipgcn_screen_{variant}_{task}"
            cmd = [
                "hpo_skipgcn.py",
                "-m",
                f"variant={variant}",
                f"task={task}",
                f"hpo.n_trials={n_trials}",
            ]
            jobs.append(
                Job(
                    key=key,
                    experiment_name=exp,
                    cmd=cmd,
                    optuna_study=study,
                )
            )
    return jobs


def _stage2_job(model: str, variant: str, task: str) -> Job:
    return Job(
        key=f"stage2/{variant}/{task}/{model}",
        experiment_name=f"hcp_stage2_compare_{variant}_{task}_{model}",
        cmd=["main_compare.py", f"model={model}", f"variant={variant}", f"task={task}"],
    )


def build_stage2_jobs() -> List[Job]:
    """
    Build stage2 jobs.

    Returns
    -------
    List[Job]
        2×4×7 compare jobs.
    """
    return [
        _stage2_job(model, variant, task)
        for variant in VARIANTS
        for task in TASKS
        for model in MODELS
    ]


def stage2_cv_experiment_name(
    variant: str,
    task: str,
    model: str,
    outer_fold: int,
) -> str:
    """
    MLflow experiment name of one stage2_cv cell.

    Outer 0 — уже посчитанные прогоны ``config_compare`` (их experiment name
    сохранён, пересчёт не нужен). Остальные fold — по шаблону
    ``config_compare_cv.yaml``.

    Parameters
    ----------
    variant : str
        Вариант графа.
    task : str
        Задача HCP.
    model : str
        Модель.
    outer_fold : int
        Номер outer fold.

    Returns
    -------
    str
        Имя MLflow experiment.
    """
    if outer_fold == 0:
        return f"hcp_stage2_compare_{variant}_{task}_{model}"
    return f"hcp_stage2_cv_{variant}_{task}_{model}_outer{outer_fold}"


def build_stage2_cv_jobs(
    model: str = "skipgcn",
    folds: Sequence[int] = OUTER_FOLDS,
    tasks: Sequence[str] = TASKS,
) -> List[Job]:
    """
    Build stage2_cv jobs: stage 2 на нескольких outer fold.

    Гиперпараметры зафиксированы: ``conf/best_task`` (подобраны на outer 0),
    одинаковые для всех fold. Порядок «задача → fold → 4 варианта»: задача
    становится пригодной для пулинга сразу после своих 4 fold × 4 вариантов.

    Parameters
    ----------
    model : str, optional
        Модель.
    folds : Sequence[int], optional
        Outer fold (0..3).
    tasks : Sequence[str], optional
        Задачи HCP.

    Returns
    -------
    List[Job]
        Jobs ``task × fold × variant``.

    Raises
    ------
    ValueError
        При неизвестных модели, fold или задаче.
    """
    if model not in MODELS:
        raise ValueError(f"model должен быть одним из {MODELS}, получено {model!r}")
    unknown_folds = sorted(set(folds) - set(OUTER_FOLDS))
    if not folds or unknown_folds:
        raise ValueError(f"folds должен быть непустым подмножеством {OUTER_FOLDS}, получено {list(folds)}")
    unknown_tasks = sorted(set(tasks) - set(TASKS))
    if not tasks or unknown_tasks:
        raise ValueError(f"tasks должен быть непустым подмножеством {TASKS}, получено {list(tasks)}")

    jobs: List[Job] = []
    for task in tasks:
        for fold in sorted(set(folds)):
            for variant in VARIANTS:
                if fold == 0:
                    cmd = ["main_compare.py", f"model={model}", f"variant={variant}", f"task={task}"]
                else:
                    cmd = [
                        "main_compare.py",
                        "--config-name",
                        "config_compare_cv",
                        f"model={model}",
                        f"variant={variant}",
                        f"task={task}",
                        f"outer_fold=fold{fold}",
                    ]
                jobs.append(
                    Job(
                        key=f"stage2_cv/{variant}/{task}/{model}/outer{fold}",
                        experiment_name=stage2_cv_experiment_name(variant, task, model, fold),
                        cmd=cmd,
                    )
                )
    return jobs


def build_leak_check_jobs(
    scope: str = "all",
    model: str = "skipgcn",
) -> List[Job]:
    """
    Build leak-check jobs.

    SkipGCN на ячейках с потолочным AUC, метки перемешаны
    (``config_leak_check.yaml``).

    Parameters
    ----------
    scope : str, optional
        ``all`` или ``train`` (``data.shuffle_labels_scope``).
    model : str, optional
        Модель.

    Returns
    -------
    List[Job]
        2×3 jobs по умолчанию.
    """
    if scope not in LEAK_CHECK_SCOPES:
        raise ValueError(f"scope должен быть одним из {LEAK_CHECK_SCOPES}, получено {scope!r}")
    if model not in MODELS:
        raise ValueError(f"model должен быть одним из {MODELS}, получено {model!r}")
    jobs: List[Job] = []
    for variant in LEAK_CHECK_VARIANTS:
        for task in LEAK_CHECK_TASKS:
            key = f"leak_check/{scope}/{variant}/{task}/{model}"
            exp = f"hcp_leak_{scope}_{variant}_{task}_{model}"
            cmd = [
                "main_compare.py",
                "--config-name",
                "config_leak_check",
                f"experiment=leak_check/shuffle_{scope}",
                f"model={model}",
                f"variant={variant}",
                f"task={task}",
            ]
            jobs.append(Job(key=key, experiment_name=exp, cmd=cmd))
    return jobs


def build_stage2_variant_jobs(model: str = "skipgcn") -> List[Job]:
    """
    Build stage2 variant-comparison jobs.

    Порядок «задача → 4 варианта»: задача становится сравнимой сразу
    после своих четырёх прогонов. Experiment names общие с ``stage2``.

    Parameters
    ----------
    model : str, optional
        Модель.

    Returns
    -------
    List[Job]
        7×4 compare jobs.
    """
    if model not in MODELS:
        raise ValueError(f"model должен быть одним из {MODELS}, получено {model!r}")
    return [_stage2_job(model, variant, task) for task in TASKS for variant in VARIANTS]


def _progress_path(batch_name: str) -> Path:
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    return PROGRESS_DIR / f"{batch_name}.json"


def save_progress(batch_name: str, jobs: Sequence[Job], start_index: int) -> None:
    """
    Save progress.

    Parameters
    ----------
    batch_name : str
        Имя batch.
    jobs : Sequence[Job]
        Список jobs.
    start_index : int
        Индекс, с которого идёт текущий запуск.
    """
    payload = {
        "batch_name": batch_name,
        "start_index": start_index,
        "jobs": [asdict(j) for j in jobs],
    }
    path = _progress_path(batch_name)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def is_job_complete(
    client,
    job: Job,
    *,
    n_trials: int,
    optuna_storage: str,
) -> bool:
    """
    Is job complete.

    Parameters
    ----------
    client : MlflowClient
        Клиент MLflow.
    job : Job
        Эксперимент.
    n_trials : int
        Целевое число trials для HPO.
    optuna_storage : str
        URI Optuna.

    Returns
    -------
    bool
        True, если эксперимент полностью завершён.
    """
    if job.optuna_study is not None:
        return optuna_study_complete(optuna_storage, job.optuna_study, n_trials)
    return has_finished_matching_parent(
        client,
        job.experiment_name,
        job.match_params,
    )


def find_resume_index(
    client,
    jobs: Sequence[Job],
    *,
    n_trials: int,
    optuna_storage: str,
) -> int:
    """
    Find resume index.

    Ищет самый длинный префикс полностью завершённых jobs
    и возвращает индекс следующего (или ``len(jobs)``, если всё готово).

    Parameters
    ----------
    client : MlflowClient
        Клиент MLflow.
    jobs : Sequence[Job]
        Упорядоченный список.
    n_trials : int
        Целевое число trials.
    optuna_storage : str
        URI Optuna.

    Returns
    -------
    int
        Индекс первого незавершённого job.
    """
    idx = 0
    while idx < len(jobs):
        job = jobs[idx]
        delete_failed_and_running(client, job.experiment_name)
        if is_job_complete(
            client,
            job,
            n_trials=n_trials,
            optuna_storage=optuna_storage,
        ):
            log.info("[%d/%d] SKIP (FINISHED) %s", idx + 1, len(jobs), job.key)
            idx += 1
            continue
        break
    return idx


def run_job(python: str, job: Job, extra_env: Optional[dict] = None) -> int:
    """
    Run job.

    Parameters
    ----------
    python : str
        Путь к интерпретатору.
    job : Job
        Эксперимент.
    extra_env : Optional[dict]
        Доп. переменные окружения.

    Returns
    -------
    int
        Код возврата процесса.
    """
    cmd = [python, *job.cmd]
    log.info("RUN: %s", " ".join(cmd))
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    return int(proc.returncode)


def run_batch(
    batch_name: str,
    jobs: Sequence[Job],
    *,
    tracking_uri: str,
    n_trials: int = 30,
    optuna_storage: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    """
    Run batch.

    Parameters
    ----------
    batch_name : str
        Имя batch (для progress-файла).
    jobs : Sequence[Job]
        Упорядоченный список экспериментов.
    tracking_uri : str
        MLflow tracking URI.
    n_trials : int, optional
        Целевое число Optuna trials.
    optuna_storage : Optional[str], optional
        URI Optuna; по умолчанию ``artifacts/optuna.db``.
    dry_run : bool, optional
        Только показать план.

    Returns
    -------
    int
        0 при успехе, иначе код ошибки.
    """
    if optuna_storage is None:
        optuna_storage = f"sqlite:///{ROOT / 'artifacts' / 'optuna.db'}"

    client = _mlflow_client(tracking_uri)
    python = _resolve_python()

    # Чистим FAILED/RUNNING/KILLED в MLflow и зависшие RUNNING в Optuna.
    for job in jobs:
        delete_failed_and_running(client, job.experiment_name)
        if job.optuna_study is not None:
            cleanup_optuna_study(optuna_storage, job.optuna_study)

    start = find_resume_index(
        client,
        jobs,
        n_trials=n_trials,
        optuna_storage=optuna_storage,
    )
    save_progress(batch_name, jobs, start)

    if start >= len(jobs):
        log.info("Все %d экспериментов уже FINISHED — нечего запускать", len(jobs))
        return 0

    log.info(
        "Возобновление batch=%s с %d/%d: %s",
        batch_name,
        start + 1,
        len(jobs),
        jobs[start].key,
    )

    if dry_run:
        for i, job in enumerate(jobs[start:], start=start):
            if job.optuna_study is not None:
                rem = optuna_remaining_trials(
                    optuna_storage,
                    job.optuna_study,
                    n_trials,
                )
                log.info(
                    "[dry-run %d/%d] %s remaining_trials=%d",
                    i + 1,
                    len(jobs),
                    job.key,
                    rem,
                )
            else:
                log.info("[dry-run %d/%d] %s", i + 1, len(jobs), job.key)
        return 0

    for i in range(start, len(jobs)):
        job = jobs[i]
        save_progress(batch_name, jobs, i)
        log.info("==== [%d/%d] %s ====", i + 1, len(jobs), job.key)

        # На случай повторного входа — ещё раз убрать FAILED/RUNNING.
        delete_failed_and_running(client, job.experiment_name)
        if job.optuna_study is not None:
            cleanup_optuna_study(optuna_storage, job.optuna_study)

        if is_job_complete(
            client,
            job,
            n_trials=n_trials,
            optuna_storage=optuna_storage,
        ):
            log.info("Уже FINISHED — skip %s", job.key)
            continue

        launch_job = job
        if job.optuna_study is not None:
            remaining = optuna_remaining_trials(
                optuna_storage,
                job.optuna_study,
                n_trials,
            )
            if remaining <= 0:
                log.info("Optuna уже набрало %d COMPLETE — skip %s", n_trials, job.key)
                continue
            launch_job = job_with_n_trials(job, remaining)
            log.info(
                "Optuna resume %s: запускаю remaining=%d (цель=%d)",
                job.optuna_study,
                remaining,
                n_trials,
            )

        code = run_job(python, launch_job)
        if code != 0:
            log.error("Эксперимент %s завершился с кодом %d", job.key, code)
            # Помечаем/чистим FAILED и останавливаем batch (resume подхватит этот же job).
            delete_failed_and_running(client, job.experiment_name)
            if job.optuna_study is not None:
                cleanup_optuna_study(optuna_storage, job.optuna_study)
            save_progress(batch_name, jobs, i)
            return code

        # Проверка, что run реально FINISHED.
        if not is_job_complete(
            client,
            job,
            n_trials=n_trials,
            optuna_storage=optuna_storage,
        ):
            log.error(
                "После запуска %s нет FINISHED-статуса — считаю незавершённым",
                job.key,
            )
            delete_failed_and_running(client, job.experiment_name)
            if job.optuna_study is not None:
                cleanup_optuna_study(optuna_storage, job.optuna_study)
            save_progress(batch_name, jobs, i)
            return 1

        log.info("FINISHED %s", job.key)
        save_progress(batch_name, jobs, i + 1)

    log.info("Batch %s полностью завершён (%d jobs)", batch_name, len(jobs))
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse args.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Аргументы CLI.

    Returns
    -------
    argparse.Namespace
        Разобранные аргументы.
    """
    parser = argparse.ArgumentParser(description="Resumable batch runner")
    parser.add_argument(
        "batch",
        choices=(
            "smoke",
            "stage0_skipgcn",
            "stage0_cfmsr",
            "confirm_A",
            "confirm_B",
            "confirm_C",
            "confirm_D",
            "stage1",
            "stage2",
            "stage2_variants",
            "stage2_cv",
            "leak_check",
        ),
        help="Тип batch-матрицы",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        metavar="K",
        help="Outer fold для stage2_cv (по умолчанию 0 1 2 3)",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        metavar="NAME",
        help="Фильтр задачи для stage2_cv (повторяемый)",
    )
    parser.add_argument(
        "--model",
        choices=MODELS,
        default="skipgcn",
        help="Модель для stage2_variants / stage2_cv / leak_check",
    )
    parser.add_argument(
        "--scope",
        choices=LEAK_CHECK_SCOPES,
        default="all",
        help="shuffle_labels_scope для leak_check",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Число Optuna trials для stage0_skipgcn / stage1",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--optuna-storage",
        default=None,
        help="Optuna storage URI",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать план возобновления",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Показать все jobs и статусы",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        metavar="NAME",
        help="Фильтр variant; ensemble = все ensemble_*",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Main.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Аргументы CLI.

    Returns
    -------
    int
        Код выхода.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args(argv)
    if args.batch != "stage2_cv" and (args.folds or args.task):
        raise SystemExit("--folds и --task поддерживаются только для batch stage2_cv")

    if args.batch == "confirm_D":
        cmd = [
            _resolve_python(),
            str(ROOT / "scripts" / "select_confirm_d.py"),
            "--tracking-uri",
            args.tracking_uri,
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        return int(subprocess.run(cmd, cwd=str(ROOT), check=False).returncode)

    if args.batch == "smoke":
        jobs = build_smoke_jobs()
    elif args.batch == "stage0_skipgcn":
        jobs = build_stage0_skipgcn_jobs(args.n_trials)
    elif args.batch == "stage0_cfmsr":
        jobs = build_stage0_cfmsr_jobs()
    elif args.batch == "confirm_A":
        jobs = build_confirm_a_jobs()
    elif args.batch == "confirm_B":
        jobs = build_confirm_b_jobs()
    elif args.batch == "confirm_C":
        jobs = build_confirm_c_jobs()
    elif args.batch == "stage1":
        jobs = build_stage1_jobs(args.n_trials)
    elif args.batch == "stage2_variants":
        jobs = build_stage2_variant_jobs(args.model)
    elif args.batch == "stage2_cv":
        jobs = build_stage2_cv_jobs(
            args.model,
            folds=tuple(args.folds) if args.folds else OUTER_FOLDS,
            tasks=tuple(args.task) if args.task else TASKS,
        )
    elif args.batch == "leak_check":
        jobs = build_leak_check_jobs(scope=args.scope, model=args.model)
    else:
        jobs = build_stage2_jobs()
    jobs = filter_jobs_by_variant(jobs, resolve_variant_filter(args.variant))

    optuna_storage = args.optuna_storage or f"sqlite:///{ROOT / 'artifacts' / 'optuna.db'}"

    if args.list:
        client = _mlflow_client(args.tracking_uri)
        for i, job in enumerate(jobs):
            delete_failed_and_running(client, job.experiment_name)
            if job.optuna_study is not None:
                cleanup_optuna_study(optuna_storage, job.optuna_study)
            done = is_job_complete(
                client,
                job,
                n_trials=args.n_trials,
                optuna_storage=optuna_storage,
            )
            status = "FINISHED" if done else "PENDING"
            extra = ""
            if job.optuna_study is not None and not done:
                rem = optuna_remaining_trials(
                    optuna_storage,
                    job.optuna_study,
                    args.n_trials,
                )
                extra = f"  remaining={rem}"
            print(f"{i + 1:3d}/{len(jobs)}  {status:8s}  {job.key}{extra}")
        return 0

    return run_batch(
        args.batch,
        jobs,
        tracking_uri=args.tracking_uri,
        n_trials=args.n_trials,
        optuna_storage=optuna_storage,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
