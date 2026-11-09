#!/usr/bin/env python3
"""
Сравнение вариантов данных для одной модели на test outer 0.

Берёт из MLflow последний FINISHED прогон ``main_compare.py`` для каждой
ячейки ``variant × task``, читает ``*test_predictions_outer0.csv`` (ансамбль
3 inner × 3 seed) и считает:

- ROC-AUC и 95% CI кластерным bootstrap по субъектам;
- парные разности ΔAUC между вариантами на одних и тех же репликах,
  95% CI и двусторонний bootstrap p-value;
- средний по задачам AUC (реплики общие, т.к. субъекты test одинаковы);
- поправку на множественность внутри двух семейств (пары с эталоном /
  остальные): Romano–Wolf step-down (основная), Holm, Benjamini–Hochberg.

Результат: CSV + markdown в ``artifacts/variant_comparison/{model}/``
и отдельный MLflow run ``hcp_variant_comparison_{model}``.
"""

from __future__ import annotations

# 1. Standard library
import argparse
import itertools
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# 2. External libraries
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# 3. Local
from batch_runner import (  # noqa: E402
    DEFAULT_TRACKING_URI,
    MODELS,
    TASKS,
    VARIANTS,
    _mlflow_client,
    _run_status,
)
from bootstrap_metrics import (  # noqa: E402
    bh_adjust,
    bootstrap_auc_replicates,
    holm_adjust,
    percentile_ci,
    romano_wolf_adjust,
    studentized_p_values,
    subject_from_stem,
)

log = logging.getLogger("compare_variants")

PREDICTIONS_SUFFIX = "test_predictions_outer0.csv"
MEAN_TASK = "MEAN"
DEFAULT_REFERENCE = "correlation"


@dataclass(frozen=True)
class TaskComparison:
    """
    Результат сравнения вариантов на одной задаче.

    Attributes
    ----------
    task : str
        Задача HCP.
    point : Dict[str, float]
        Точечный AUC по вариантам.
    replicates : Dict[str, np.ndarray]
        Bootstrap-реплики AUC по вариантам (общие индексы).
    subjects : Tuple[str, ...]
        Отсортированные субъекты test.
    """

    task: str
    point: Dict[str, float]
    replicates: Dict[str, np.ndarray]
    subjects: Tuple[str, ...]


def align_predictions(preds: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Align predictions by stem.

    Parameters
    ----------
    preds : Dict[str, pd.DataFrame]
        Предсказания по вариантам: колонки ``stem``, ``prob``, ``label``.

    Returns
    -------
    pd.DataFrame
        Колонки ``stem``, ``subject``, ``label`` и по колонке на вариант.

    Raises
    ------
    ValueError
        Если наборы образцов или метки различаются между вариантами.
    """
    if not preds:
        raise ValueError("preds пуст")
    merged: Optional[pd.DataFrame] = None
    for variant, df in preds.items():
        missing = {"stem", "prob", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"{variant}: нет колонок {sorted(missing)}")
        if df["stem"].duplicated().any():
            raise ValueError(f"{variant}: дубликаты stem в предсказаниях")
        part = df[["stem", "label", "prob"]].rename(
            columns={"prob": variant, "label": f"label_{variant}"},
        )
        if merged is None:
            merged = part
            continue
        if set(part["stem"]) != set(merged["stem"]):
            raise ValueError(f"{variant}: набор test-образцов отличается от остальных вариантов")
        merged = merged.merge(part, on="stem", how="inner", validate="one_to_one")

    assert merged is not None
    label_cols = [f"label_{v}" for v in preds]
    labels = merged[label_cols].to_numpy(dtype=np.float64)
    if not np.all(labels == labels[:, :1]):
        raise ValueError("Метки test различаются между вариантами")
    merged["label"] = labels[:, 0].astype(int)
    merged["subject"] = merged["stem"].map(subject_from_stem)
    merged = merged.drop(columns=label_cols).sort_values("stem").reset_index(drop=True)
    return merged[["stem", "subject", "label", *preds.keys()]]


def compare_task(
    task: str,
    preds: Dict[str, pd.DataFrame],
    n_bootstrap: int,
    seed: int,
) -> TaskComparison:
    """
    Compare variants on one task.

    Parameters
    ----------
    task : str
        Задача HCP.
    preds : Dict[str, pd.DataFrame]
        Предсказания по вариантам.
    n_bootstrap : int
        Число bootstrap-реплик.
    seed : int
        Random seed (общий для всех задач → общие реплики субъектов).

    Returns
    -------
    TaskComparison
        Точечные AUC и реплики.
    """
    aligned = align_predictions(preds)
    point, reps = bootstrap_auc_replicates(
        aligned["label"].to_numpy(),
        {v: aligned[v].to_numpy() for v in preds},
        groups=aligned["subject"].tolist(),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return TaskComparison(
        task=task,
        point=point,
        replicates=reps,
        subjects=tuple(sorted(set(aligned["subject"]))),
    )


def mean_over_tasks(results: Sequence[TaskComparison]) -> TaskComparison:
    """
    Mean AUC over tasks.

    Parameters
    ----------
    results : Sequence[TaskComparison]
        Сравнения по задачам с одинаковым набором вариантов.

    Returns
    -------
    TaskComparison
        Средний AUC и реплики среднего (``task=MEAN``).
    """
    if not results:
        raise ValueError("results пуст")
    variants = list(results[0].point)
    if any(list(r.point) != variants for r in results):
        raise ValueError("Набор вариантов различается между задачами")
    if len({r.subjects for r in results}) > 1:
        log.warning("Субъекты test различаются между задачами: CI среднего приближённый")
    point = {v: float(np.mean([r.point[v] for r in results])) for v in variants}
    reps = {v: np.mean(np.vstack([r.replicates[v] for r in results]), axis=0) for v in variants}
    return TaskComparison(task=MEAN_TASK, point=point, replicates=reps, subjects=results[0].subjects)


def auc_table(results: Sequence[TaskComparison], alpha: float = 0.05) -> pd.DataFrame:
    """
    AUC table with CI.

    Parameters
    ----------
    results : Sequence[TaskComparison]
        Сравнения по задачам (включая MEAN).
    alpha : float, optional
        Уровень CI.

    Returns
    -------
    pd.DataFrame
        ``task, variant, roc_auc, ci_low, ci_high, rank``.
    """
    rows = []
    for res in results:
        ranks = pd.Series(res.point).rank(ascending=False, method="min")
        for variant, auc in res.point.items():
            low, high = percentile_ci(res.replicates[variant], alpha=alpha)
            rows.append(
                {
                    "task": res.task,
                    "variant": variant,
                    "roc_auc": auc,
                    "ci_low": low,
                    "ci_high": high,
                    "rank": int(ranks[variant]),
                }
            )
    return pd.DataFrame(rows)


def _ordered_pairs(variants: Sequence[str], reference: Optional[str]) -> List[Tuple[str, str]]:
    """Пары вариантов; в парах с эталоном он всегда ``a`` (ΔAUC = другой − эталон)."""
    pairs = []
    for var_a, var_b in itertools.combinations(variants, 2):
        if var_b == reference:
            var_a, var_b = var_b, var_a
        pairs.append((var_a, var_b))
    return pairs


def pairwise_table(
    results: Sequence[TaskComparison],
    reference: Optional[str] = DEFAULT_REFERENCE,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Pairwise ΔAUC table with multiplicity adjustment.

    ``delta = AUC(variant_b) − AUC(variant_a)``; ``p_value`` — сырой
    студентизированный bootstrap p (общий для всех поправок). Гипотезы делятся на два
    заранее заданных семейства: ``primary`` — пары с эталоном ``reference``,
    ``secondary`` — остальные; в оба входят строки всех задач и MEAN.
    Внутри семейства считаются:

    - ``p_romano_wolf`` — основная поправка (FWER, учитывает зависимость
      через совместные реплики);
    - ``p_holm`` — FWER без допущений, проверка устойчивости;
    - ``p_bh`` — FDR (Benjamini-Hochberg), разведочно.

    Parameters
    ----------
    results : Sequence[TaskComparison]
        Сравнения по задачам (включая MEAN) с общими репликами.
    reference : Optional[str], optional
        Эталонный вариант; ``None`` — все пары в одном семействе ``primary``.
    alpha : float, optional
        Уровень CI.

    Returns
    -------
    pd.DataFrame
        ``family, task, variant_a, variant_b, delta, ci_low, ci_high,
        p_value, p_romano_wolf, p_holm, p_bh``.
    """
    if not results:
        raise ValueError("results пуст")
    variants = list(results[0].point)
    if reference is not None and reference not in variants:
        raise ValueError(f"reference={reference!r} нет среди вариантов {variants}")
    n_boot = {r.replicates[variants[0]].size for r in results}
    if len(n_boot) != 1:
        raise ValueError(f"Разное число реплик между задачами: {sorted(n_boot)}")

    rows = []
    diffs: List[np.ndarray] = []
    for res in results:
        for var_a, var_b in _ordered_pairs(variants, reference):
            diff = res.replicates[var_b] - res.replicates[var_a]
            low, high = percentile_ci(diff, alpha=alpha)
            is_primary = reference is None or reference in (var_a, var_b)
            rows.append(
                {
                    "family": "primary" if is_primary else "secondary",
                    "task": res.task,
                    "variant_a": var_a,
                    "variant_b": var_b,
                    "delta": res.point[var_b] - res.point[var_a],
                    "ci_low": low,
                    "ci_high": high,
                }
            )
            diffs.append(diff)

    table = pd.DataFrame(rows)
    diff_matrix = np.column_stack(diffs)
    table["p_value"] = studentized_p_values(table["delta"].to_numpy(), diff_matrix)
    for col in ("p_romano_wolf", "p_holm", "p_bh"):
        table[col] = np.nan
    for family in table["family"].unique():
        idx = np.flatnonzero(table["family"].to_numpy() == family)
        table.loc[idx, "p_romano_wolf"] = romano_wolf_adjust(
            table.loc[idx, "delta"].to_numpy(),
            diff_matrix[:, idx],
        )
        raw = table.loc[idx, "p_value"].tolist()
        table.loc[idx, "p_holm"] = holm_adjust(raw)
        table.loc[idx, "p_bh"] = bh_adjust(raw)
    return table


def _verdict(p_romano_wolf: float, p_holm: float, alpha: float) -> str:
    """Вывод по строке: значимость по RW и устойчивость к Holm."""
    if not p_romano_wolf < alpha:
        return ""
    return "RW+Holm" if p_holm < alpha else "RW"


def _fmt_ci(value: float, low: float, high: float, signed: bool = False) -> str:
    sign = "+" if signed else ""
    return f"{value:{sign}.4f} [{low:{sign}.4f}, {high:{sign}.4f}]"


def render_markdown(
    model: str,
    aucs: pd.DataFrame,
    pairs: pd.DataFrame,
    n_bootstrap: int,
    alpha: float = 0.05,
    title: Optional[str] = None,
) -> str:
    """
    Render markdown report.

    Parameters
    ----------
    model : str
        Модель.
    aucs : pd.DataFrame
        Таблица :func:`auc_table`.
    pairs : pd.DataFrame
        Таблица :func:`pairwise_table`.
    n_bootstrap : int
        Число реплик (для подписи).
    alpha : float, optional
        Уровень значимости для колонки «вывод».
    title : Optional[str], optional
        Заголовок первого уровня; по умолчанию «Сравнение вариантов: {model}, test outer 0».

    Returns
    -------
    str
        Markdown-отчёт.
    """
    variants = list(dict.fromkeys(aucs["variant"]))
    tasks = list(dict.fromkeys(aucs["task"]))
    lines = [
        f"# {title or f'Сравнение вариантов: {model}, test outer 0'}",
        "",
        f"ROC-AUC [95% CI], кластерный bootstrap по субъектам, B={n_bootstrap}. "
        "Жирным — лучший вариант задачи.",
        "",
        "| task | " + " | ".join(variants) + " |",
        "|---|" + "---|" * len(variants),
    ]
    for task in tasks:
        sub = aucs[aucs["task"] == task].set_index("variant")
        cells = []
        for v in variants:
            row = sub.loc[v]
            text = _fmt_ci(row["roc_auc"], row["ci_low"], row["ci_high"])
            cells.append(f"**{text}**" if row["rank"] == 1 else text)
        lines.append(f"| {task} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "ΔAUC = b − a [95% CI]. Поправки внутри семейства (все задачи + MEAN): "
        "p_RW — Romano–Wolf step-down (основная, FWER), p_Holm — проверка устойчивости, "
        f"q_BH — FDR (разведочно). Вывод: `RW+Holm` — значимо по обеим при α={alpha}, "
        "`RW` — только по Romano–Wolf.",
    ]
    titles = {
        "primary": "Основное семейство (сравнения с эталоном)",
        "secondary": "Вторичное семейство (остальные пары)",
    }
    for family in pairs["family"].unique():
        sub = pairs[pairs["family"] == family]
        lines += [
            "",
            f"## {titles.get(family, family)} — гипотез: {len(sub)}",
            "",
            "| task | a | b | ΔAUC [95% CI] | p | p_RW | p_Holm | q_BH | вывод |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for row in sub.itertuples(index=False):
            lines.append(
                f"| {row.task} | {row.variant_a} | {row.variant_b} | "
                f"{_fmt_ci(row.delta, row.ci_low, row.ci_high, signed=True)} | "
                f"{row.p_value:.4f} | {row.p_romano_wolf:.4f} | {row.p_holm:.4f} | "
                f"{row.p_bh:.4f} | {_verdict(row.p_romano_wolf, row.p_holm, alpha)} |"
            )
    return "\n".join(lines) + "\n"


def _latest_finished_run(client, experiment_name: str) -> Optional[object]:
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=100,
    )
    for run in runs:
        if _run_status(run) == "FINISHED" and run.data.tags.get("mlflow.parentRunId") is None:
            return run
    return None


def load_experiment_predictions(
    client,
    experiment_name: str,
    suffix: str,
    dst: Path,
) -> Optional[Tuple[pd.DataFrame, str]]:
    """
    Load test predictions of the latest FINISHED run of an experiment.

    Parameters
    ----------
    client : MlflowClient
        MLflow client.
    experiment_name : str
        Имя MLflow experiment.
    suffix : str
        Суффикс артефакта предсказаний (``*test_predictions_outer0.csv``).
    dst : Path
        Каталог для скачивания артефакта.

    Returns
    -------
    Optional[Tuple[pd.DataFrame, str]]
        Предсказания и ``run_id``; ``None`` — нет FINISHED run или артефакта.
    """
    run = _latest_finished_run(client, experiment_name)
    if run is None:
        return None
    # Имя зависит от training.predictions_out (``{model}_test_predictions_outer0.csv``).
    names = [a.path for a in client.list_artifacts(run.info.run_id) if a.path.endswith(suffix)]
    if len(names) != 1:
        log.warning("run %s: ожидался один *%s, найдено %s", run.info.run_id, suffix, names)
        return None
    dst.mkdir(parents=True, exist_ok=True)
    path = client.download_artifacts(run.info.run_id, names[0], str(dst))
    return pd.read_csv(path), run.info.run_id


def load_task_predictions(
    client,
    model: str,
    task: str,
    variants: Sequence[str],
    tmp_dir: Path,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """
    Load test predictions of one task from MLflow.

    Parameters
    ----------
    client : MlflowClient
        MLflow client.
    model : str
        Модель.
    task : str
        Задача.
    variants : Sequence[str]
        Варианты данных.
    tmp_dir : Path
        Каталог для скачивания артефактов.

    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], Dict[str, str]]
        Предсказания и run_id по вариантам (только найденные).
    """
    preds: Dict[str, pd.DataFrame] = {}
    run_ids: Dict[str, str] = {}
    for variant in variants:
        loaded = load_experiment_predictions(
            client,
            f"hcp_stage2_compare_{variant}_{task}_{model}",
            PREDICTIONS_SUFFIX,
            tmp_dir / f"{variant}_{task}",
        )
        if loaded is not None:
            preds[variant], run_ids[variant] = loaded
    return preds, run_ids


def _log_to_mlflow(
    tracking_uri: str,
    model: str,
    aucs: pd.DataFrame,
    pairs: pd.DataFrame,
    out_dir: Path,
    params: Dict[str, object],
    run_ids: Dict[str, Dict[str, str]],
) -> None:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"hcp_variant_comparison_{model}")
    alpha = float(params["alpha"])  # type: ignore[arg-type]
    with mlflow.start_run(run_name=f"variants_{model}_outer0"):
        mlflow.set_tags({"model": model, "stage": "stage2_variants", "outer_fold": "0"})
        mlflow.log_params({**params, "bootstrap": "cluster_subject", "fwer": "romano_wolf"})
        for family, sub in pairs.groupby("family"):
            mlflow.log_metrics(
                {
                    f"{family}_n_hypotheses": float(len(sub)),
                    f"{family}_n_sig_romano_wolf": float((sub["p_romano_wolf"] < alpha).sum()),
                    f"{family}_n_sig_holm": float((sub["p_holm"] < alpha).sum()),
                    f"{family}_n_sig_bh": float((sub["p_bh"] < alpha).sum()),
                }
            )
        for task, ids in run_ids.items():
            for variant, run_id in ids.items():
                mlflow.set_tag(f"source_{task}_{variant}", run_id)
        for row in aucs.itertuples(index=False):
            prefix = f"{row.task}_{row.variant}"
            mlflow.log_metrics(
                {
                    f"{prefix}_roc_auc": float(row.roc_auc),
                    f"{prefix}_ci_low": float(row.ci_low),
                    f"{prefix}_ci_high": float(row.ci_high),
                }
            )
        mlflow.log_artifacts(str(out_dir))


def run_compare_variants(
    model: str,
    tracking_uri: str,
    out_dir: Path,
    n_bootstrap: int = 10000,
    seed: int = 42,
    reference: Optional[str] = DEFAULT_REFERENCE,
    alpha: float = 0.05,
    log_mlflow: bool = True,
) -> int:
    """
    Run variant comparison.

    Parameters
    ----------
    model : str
        Модель.
    tracking_uri : str
        MLflow tracking URI.
    out_dir : Path
        Каталог отчёта.
    n_bootstrap : int, optional
        Число реплик.
    seed : int, optional
        Random seed.
    reference : Optional[str], optional
        Эталонный вариант основного семейства; ``None`` — одно семейство.
    alpha : float, optional
        Уровень значимости.
    log_mlflow : bool, optional
        Писать ли отдельный MLflow run.

    Returns
    -------
    int
        0 — успех, 1 — нет ни одной полной задачи.
    """
    if model not in MODELS:
        raise ValueError(f"model должен быть одним из {MODELS}, получено {model!r}")
    if n_bootstrap < 100:
        raise ValueError(f"n_bootstrap слишком мал: {n_bootstrap}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha должен быть в (0, 1), получено {alpha}")
    if reference is not None and reference not in VARIANTS:
        raise ValueError(f"reference должен быть одним из {VARIANTS}, получено {reference!r}")

    client = _mlflow_client(tracking_uri)
    results: List[TaskComparison] = []
    run_ids: Dict[str, Dict[str, str]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for task in TASKS:
            preds, ids = load_task_predictions(client, model, task, VARIANTS, Path(tmp))
            if len(preds) < len(VARIANTS):
                missing = sorted(set(VARIANTS) - set(preds))
                log.warning("Задача %s пропущена: нет прогонов %s", task, missing)
                continue
            results.append(compare_task(task, preds, n_bootstrap=n_bootstrap, seed=seed))
            run_ids[task] = ids

    if not results:
        log.error("Нет ни одной задачи со всеми %d вариантами", len(VARIANTS))
        return 1
    if len(results) > 1:
        results.append(mean_over_tasks(results))

    aucs = auc_table(results)
    pairs = pairwise_table(results, reference=reference)
    report = render_markdown(model, aucs, pairs, n_bootstrap, alpha=alpha)

    out_dir.mkdir(parents=True, exist_ok=True)
    aucs.to_csv(out_dir / "auc_by_task.csv", index=False)
    pairs.to_csv(out_dir / "pairwise_delta.csv", index=False)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    log.info("Отчёт записан в %s", out_dir)

    if log_mlflow:
        params: Dict[str, object] = {
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "reference": str(reference),
            "alpha": alpha,
        }
        _log_to_mlflow(tracking_uri, model, aucs, pairs, out_dir, params, run_ids)
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse CLI arguments.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Аргументы CLI.

    Returns
    -------
    argparse.Namespace
        Разобранные аргументы.
    """
    parser = argparse.ArgumentParser(description="Сравнение вариантов данных на test outer 0")
    parser.add_argument("--model", choices=MODELS, default="skipgcn")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE,
        help="Эталон основного семейства гипотез; 'none' — все пары в одном семействе",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-mlflow", action="store_true", help="Не создавать MLflow run")
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)
    out_dir = args.out_dir or ROOT / "artifacts" / "variant_comparison" / args.model
    return run_compare_variants(
        model=args.model,
        tracking_uri=args.tracking_uri,
        out_dir=Path(out_dir).resolve(),
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        reference=None if args.reference.lower() == "none" else args.reference,
        alpha=args.alpha,
        log_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    raise SystemExit(main())
