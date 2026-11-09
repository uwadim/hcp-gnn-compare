#!/usr/bin/env python3
"""
Классификатор «средний вес top-n рёбер» и зависимость ROC-AUC от n.

У каждого графа берутся n рёбер с наибольшим ``|вес|`` (верхний треугольник
без диагонали), скор — среднее их знаковых весов. Знак скора выбирается по
train pool для каждого n отдельно, затем применяется к outer test.
``n = все рёбра`` совпадает с классификатором ``mean_edge_baseline.py``.

Для каждого варианта данных (4) и задачи (7) строится кривая ROC-AUC(n)
с 95% CI (кластерный bootstrap по субъектам, реплики общие для всех n).
Графики: по одному на вариант (линии — задачи) и сводный по среднему AUC
по задачам (линии — варианты).

Несколько outer fold (``--outer-fold 0 1 2 3``, по умолчанию все): знак выбирается по train pool
своего fold, test-скоры всех fold объединяются, CI — один bootstrap по субъектам всех fold.

Результат: ``artifacts/top_n_edge_baseline/`` — ``auc_by_n.csv``,
``auc_by_n_mean.csv``, ``best_n_by_train.csv``, ``report.md``, ``*.png`` и ``*.svg``
(английские подписи — файлы с суффиксом ``_en``, параметр ``--plot-lang``).
"""

from __future__ import annotations

# 1. Standard library
import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple, Union

# 2. External libraries
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# 3. Local
from batch_runner import OUTER_FOLDS, TASKS, VARIANTS  # noqa: E402
from bootstrap_metrics import bootstrap_auc_replicates, percentile_ci  # noqa: E402
from compare_variants import TaskComparison, mean_over_tasks  # noqa: E402
from compare_variants_cv import expected_test_subjects, normalize_folds, validate_fold_subjects  # noqa: E402
from config_utils import resolve_existing_path  # noqa: E402
from dataset import GraphSample  # noqa: E402
from mean_edge_baseline import _upper_indices, fit_sign, is_degenerate, load_samples  # noqa: E402

log = logging.getLogger("top_n_edge_baseline")

DEFAULT_N_VALUES = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000)

DEFAULT_LANG = "ru"
FIGURE_FORMATS = ("png", "svg")


@dataclass(frozen=True)
class PlotLabels:
    """Подписи графиков на одном языке."""

    xlabel: str
    ylabel: str
    title_variant: str  # поля: {variant}, {fold}
    title_mean: str  # поле: {fold}
    legend_tasks: str
    legend_variants: str
    file_suffix: str


PLOT_LABELS: Mapping[str, PlotLabels] = {
    "ru": PlotLabels(
        xlabel="n (число рёбер с наибольшим |весом|)",
        ylabel="ROC-AUC на test",
        title_variant="{variant}: ROC-AUC vs top-n рёбер (test outer {fold}, 95% CI)",
        title_mean="Средний по задачам ROC-AUC vs top-n рёбер (test outer {fold}, 95% CI)",
        legend_tasks="Задача",
        legend_variants="Вариант",
        file_suffix="",
    ),
    "en": PlotLabels(
        xlabel="n (number of edges with the largest |weight|)",
        ylabel="Test ROC-AUC",
        title_variant="{variant}: ROC-AUC vs top-n edges (test outer {fold}, 95% CI)",
        title_mean="Task-averaged ROC-AUC vs top-n edges (test outer {fold}, 95% CI)",
        legend_tasks="Task",
        legend_variants="Variant",
        file_suffix="_en",
    ),
}
PLOT_LANG_CHOICES = (*PLOT_LABELS, "both")


def resolve_langs(plot_lang: str) -> Tuple[str, ...]:
    """
    Expand ``--plot-lang`` value into a tuple of language codes.

    Parameters
    ----------
    plot_lang : str
        ``ru``, ``en`` или ``both``.

    Returns
    -------
    Tuple[str, ...]
        Коды языков для построения графиков.
    """
    if plot_lang == "both":
        return tuple(PLOT_LABELS)
    if plot_lang not in PLOT_LABELS:
        raise ValueError(f"Неизвестный язык {plot_lang!r}, допустимы {PLOT_LANG_CHOICES}")
    return (plot_lang,)
META_COLUMNS = ("stem", "subject", "label", "role")
MEAN_TASK = "MEAN"


def build_n_grid(n_edges: int, values: Sequence[int] = DEFAULT_N_VALUES) -> List[int]:
    """
    Build n grid.

    Parameters
    ----------
    n_edges : int
        Число рёбер графа (верхний треугольник).
    values : Sequence[int], optional
        Значения n; большие ``n_edges`` отбрасываются.

    Returns
    -------
    List[int]
        Отсортированная сетка без повторов, всегда заканчивается на ``n_edges``.

    Raises
    ------
    ValueError
        Если ``n_edges < 1`` или есть значения < 1.
    """
    if n_edges < 1:
        raise ValueError(f"n_edges должен быть ≥ 1, получено {n_edges}")
    if any(int(v) < 1 for v in values):
        raise ValueError(f"Все n должны быть ≥ 1, получено {list(values)}")
    grid = sorted({int(v) for v in values if int(v) < n_edges} | {int(n_edges)})
    return grid


def top_n_means(matrix: np.ndarray, ns: Sequence[int]) -> np.ndarray:
    """
    Mean of top-n edges by absolute weight.

    Parameters
    ----------
    matrix : np.ndarray
        Квадратная матрица весов ``(N, N)``.
    ns : Sequence[int]
        Значения n (``1 ≤ n ≤ N(N−1)/2``).

    Returns
    -------
    np.ndarray
        Для каждого n среднее знаковых весов n рёбер с наибольшим ``|w|``.

    Raises
    ------
    ValueError
        Если матрица не квадратная или n вне допустимого диапазона.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError(f"Ожидалась квадратная матрица N≥2, shape={matrix.shape}")
    weights = matrix[_upper_indices(matrix.shape[0])].astype(np.float64)
    ns_np = np.asarray(ns, dtype=np.int64)
    if ns_np.size == 0 or ns_np.min() < 1 or ns_np.max() > weights.size:
        raise ValueError(f"n должен быть в [1, {weights.size}], получено {ns_np.tolist()}")
    order = np.argsort(-np.abs(weights), kind="stable")
    cumulative = np.cumsum(weights[order])
    return cumulative[ns_np - 1] / ns_np


def score_samples_top_n(
    samples: Sequence[GraphSample],
    ns: Sequence[int],
    workers: int,
) -> np.ndarray:
    """
    Score samples for all n.

    Parameters
    ----------
    samples : Sequence[GraphSample]
        Образцы с путями к матрицам рёбер.
    ns : Sequence[int]
        Значения n.
    workers : int
        Число потоков чтения.

    Returns
    -------
    np.ndarray
        Матрица ``(len(samples), len(ns))``.
    """
    if workers < 1:
        raise ValueError(f"workers должен быть ≥ 1, получено {workers}")

    def _one(sample: GraphSample) -> np.ndarray:
        return top_n_means(np.load(sample.edge_path), ns)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_one, samples))
    return np.vstack(rows)


def _score_columns(ns: Sequence[int]) -> List[str]:
    return [f"top{n}" for n in ns]


def collect_scores(
    data_root: Path,
    variant: str,
    task: str,
    outer_fold: int,
    ns: Sequence[int],
    workers: int,
    cache_dir: Optional[Path],
) -> pd.DataFrame:
    """
    Collect top-n scores of train pool and test.

    Parameters
    ----------
    data_root : Path
        Корень архива HCP.
    variant : str
        Вариант данных.
    task : str
        Задача HCP.
    outer_fold : int
        Outer fold.
    ns : Sequence[int]
        Сетка n.
    workers : int
        Число потоков чтения.
    cache_dir : Optional[Path]
        Каталог кэша; ``None`` — без кэша. Кэш с другой сеткой n пересчитывается.

    Returns
    -------
    pd.DataFrame
        ``stem, subject, label, role`` и колонки ``top{n}``.
    """
    columns = [*META_COLUMNS, *_score_columns(ns)]
    cache = None
    if cache_dir is not None:
        cache = cache_dir / f"{variant}_{task}_outer{outer_fold}.csv"
        if cache.is_file():
            frame = pd.read_csv(cache, dtype={"stem": str, "subject": str})
            if list(frame.columns) == columns:
                return frame
            log.warning("Кэш %s с другой сеткой n — пересчёт", cache)

    pool_samples, test_samples = load_samples(data_root, variant, task, outer_fold)
    parts = []
    for role, samples in (("train", pool_samples), ("test", test_samples)):
        part = pd.DataFrame(
            {
                "stem": [s.stem for s in samples],
                "subject": [s.subject for s in samples],
                "label": [float(s.label) for s in samples],
                "role": role,
            }
        )
        scores = score_samples_top_n(samples, ns, workers)
        part[_score_columns(ns)] = scores
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache, index=False)
    return frame


def evaluate_cell(
    frame: pd.DataFrame,
    variant: str,
    task: str,
    ns: Sequence[int],
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, TaskComparison]:
    """
    Evaluate AUC(n) curve for one variant × task on one outer fold.

    Parameters
    ----------
    frame : pd.DataFrame
        Результат :func:`collect_scores`.
    variant : str
        Вариант данных.
    task : str
        Задача HCP.
    ns : Sequence[int]
        Сетка n.
    n_bootstrap : int
        Число bootstrap-реплик.
    seed : int
        Random seed (общий для всех ячеек → общие реплики субъектов).

    Returns
    -------
    tuple[pd.DataFrame, TaskComparison]
        См. :func:`evaluate_cell_folds`.
    """
    return evaluate_cell_folds([frame], variant, task, ns, n_bootstrap, seed)


def evaluate_cell_folds(
    frames: Sequence[pd.DataFrame],
    variant: str,
    task: str,
    ns: Sequence[int],
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, TaskComparison]:
    """
    Evaluate pooled AUC(n) curve for one variant × task over several outer folds.

    Знак скора для каждого n выбирается по train pool своего fold, затем
    знаковые test-скоры всех fold объединяются, CI — один кластерный bootstrap
    по субъектам всех fold (каждый субъект в test ровно одного fold).

    Parameters
    ----------
    frames : Sequence[pd.DataFrame]
        Результаты :func:`collect_scores`, по одному на outer fold.
    variant : str
        Вариант данных.
    task : str
        Задача HCP.
    ns : Sequence[int]
        Сетка n.
    n_bootstrap : int
        Число bootstrap-реплик.
    seed : int
        Random seed (общий для всех ячеек → общие реплики субъектов).

    Returns
    -------
    tuple[pd.DataFrame, TaskComparison]
        Таблица ``variant, task, n, sign, train_auc, roc_auc, ci_low, ci_high,
        degenerate`` и реплики AUC по n (для усреднения по задачам). ``sign`` —
        общий знак по fold (0, если fold расходятся), ``train_auc`` — среднее по
        fold, ``degenerate`` — вырожден скор хотя бы в одном fold.
    """
    if not frames:
        raise ValueError("frames пуст")
    signs: Dict[int, List[int]] = {n: [] for n in ns}
    train_auc: Dict[int, List[float]] = {n: [] for n in ns}
    degenerate: Dict[int, bool] = {n: False for n in ns}
    signed_tests: List[pd.DataFrame] = []
    for frame in frames:
        train = frame[frame["role"] == "train"]
        test = frame[frame["role"] == "test"]
        if train.empty or test.empty:
            raise ValueError(f"{variant}/{task}: пустой train или test")
        y_train = train["label"].to_numpy()
        signed = test[["label", "subject"]].copy()
        for n in ns:
            col = f"top{n}"
            sign, auc = fit_sign(y_train, train[col].to_numpy())
            signs[n].append(sign)
            train_auc[n].append(auc)
            degenerate[n] = degenerate[n] or is_degenerate(frame[col].to_numpy())
            signed[str(n)] = sign * test[col].to_numpy()
        signed_tests.append(signed)

    pooled = pd.concat(signed_tests, ignore_index=True)
    point, reps = bootstrap_auc_replicates(
        pooled["label"].to_numpy(),
        {str(n): pooled[str(n)].to_numpy() for n in ns},
        groups=pooled["subject"].astype(str).tolist(),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    rows = []
    for n in ns:
        low, high = percentile_ci(reps[str(n)])
        rows.append(
            {
                "variant": variant,
                "task": task,
                "n": n,
                "sign": signs[n][0] if len(set(signs[n])) == 1 else 0,
                "train_auc": float(pd.Series(train_auc[n]).mean()),
                "roc_auc": point[str(n)],
                "ci_low": low,
                "ci_high": high,
                "degenerate": degenerate[n],
            }
        )
    comparison = TaskComparison(
        task=task,
        point=point,
        replicates=reps,
        subjects=tuple(sorted(set(pooled["subject"].astype(str)))),
    )
    return pd.DataFrame(rows), comparison


def fold_label(folds: Sequence[int]) -> str:
    """
    Label of a fold set for titles.

    Parameters
    ----------
    folds : Sequence[int]
        Outer fold.

    Returns
    -------
    str
        ``"0"`` для одного fold, ``"0, 1, 2, 3 (pooled)"`` для нескольких.
    """
    listed = ", ".join(str(k) for k in folds)
    return listed if len(folds) == 1 else f"{listed} (pooled)"


def mean_curve(variant: str, comparisons: Sequence[TaskComparison]) -> pd.DataFrame:
    """
    Mean AUC(n) over tasks.

    Parameters
    ----------
    variant : str
        Вариант данных.
    comparisons : Sequence[TaskComparison]
        Реплики по задачам одного варианта.

    Returns
    -------
    pd.DataFrame
        ``variant, task=MEAN, n, roc_auc, ci_low, ci_high``.
    """
    mean = mean_over_tasks(comparisons)
    rows = []
    for key, auc in mean.point.items():
        low, high = percentile_ci(mean.replicates[key])
        rows.append(
            {
                "variant": variant,
                "task": MEAN_TASK,
                "n": int(key),
                "roc_auc": auc,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def best_n_by_train(curves: pd.DataFrame) -> pd.DataFrame:
    """
    Best n selected on train AUC.

    n выбирается по train AUC (test не участвует); при равенстве — наименьший n.
    Вырожденные скоры исключаются.

    Parameters
    ----------
    curves : pd.DataFrame
        Таблица кривых по ячейкам.

    Returns
    -------
    pd.DataFrame
        По строке на variant × task: выбранный n, train AUC и test AUC [CI].
    """
    rows = []
    valid = curves[~curves["degenerate"] & curves["train_auc"].notna()]
    for (variant, task), group in valid.groupby(["variant", "task"], sort=False):
        best = group.sort_values(["train_auc", "n"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "variant": variant,
                "task": task,
                "best_n": int(best["n"]),
                "train_auc": best["train_auc"],
                "roc_auc": best["roc_auc"],
                "ci_low": best["ci_low"],
                "ci_high": best["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def _xtick_labels(ns: Sequence[int], n_edges: int) -> List[str]:
    return ["all" if n == n_edges else str(n) for n in ns]


def ylim_for_curves(
    y_low: np.ndarray,
    y_high: np.ndarray,
    *,
    fill_ratio: float = 0.75,
    min_span: float = 1e-3,
) -> Tuple[float, float]:
    """
    Compute y-limits so data span occupies at least ``fill_ratio`` of the axis.

    Parameters
    ----------
    y_low : np.ndarray
        Нижние границы данных (например ci_low).
    y_high : np.ndarray
        Верхние границы данных (например ci_high).
    fill_ratio : float
        Минимальная доля высоты оси, которую должен занимать размах данных.
    min_span : float
        Нижняя граница размаха данных при вырожденных кривых.

    Returns
    -------
    Tuple[float, float]
        ``(ymin, ymax)`` для ``Axes.set_ylim``.
    """
    if not 0.0 < fill_ratio <= 1.0:
        raise ValueError(f"fill_ratio должен быть в (0, 1], получено {fill_ratio}")
    if min_span <= 0.0:
        raise ValueError(f"min_span должен быть > 0, получено {min_span}")
    low = float(np.nanmin(y_low))
    high = float(np.nanmax(y_high))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("y_low/y_high содержат только NaN/Inf")
    data_span = max(high - low, min_span)
    axis_span = data_span / fill_ratio
    pad = 0.5 * (axis_span - data_span)
    center = 0.5 * (low + high)
    if high - low < min_span:
        return center - 0.5 * axis_span, center + 0.5 * axis_span
    return low - pad, high + pad


def plot_curves(
    curves: pd.DataFrame,
    mean_curves: pd.DataFrame,
    out_dir: Path,
    outer_fold: Union[int, str],
    lang: str = DEFAULT_LANG,
) -> List[Path]:
    """
    Plot AUC vs n.

    Parameters
    ----------
    curves : pd.DataFrame
        Таблица кривых по ячейкам variant × task.
    mean_curves : pd.DataFrame
        Таблица средних по задачам кривых.
    out_dir : Path
        Каталог для рисунков.
    outer_fold : Union[int, str]
        Outer fold или подпись набора fold (для заголовков).
    lang : str, optional
        Язык подписей осей, заголовка и легенды: ``ru`` или ``en``.

    Returns
    -------
    List[Path]
        Пути созданных файлов (PNG и SVG): по одному рисунку на вариант и сводный.
    """
    if lang not in PLOT_LABELS:
        raise ValueError(f"Неизвестный язык {lang!r}, допустимы {tuple(PLOT_LABELS)}")
    labels = PLOT_LABELS[lang]
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    ns = sorted(curves["n"].unique())
    n_edges = max(ns)
    ticks = _xtick_labels(ns, n_edges)
    paths: List[Path] = []

    def _style(ax: "plt.Axes", title: str, y_low: np.ndarray, y_high: np.ndarray) -> None:
        ymin, ymax = ylim_for_curves(y_low, y_high)
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels(ticks, rotation=45, ha="right")
        ax.minorticks_off()
        if ymin < 0.5 < ymax:
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel(labels.xlabel)
        ax.set_ylabel(labels.ylabel)
        ax.set_ylim(ymin, ymax)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    for variant in curves["variant"].unique():
        fig, ax = plt.subplots(figsize=(9, 5.5))
        subset = curves[curves["variant"] == variant]
        for task, group in subset.groupby("task", sort=False):
            group = group.sort_values("n")
            ax.plot(group["n"], group["roc_auc"], marker="o", markersize=3, label=task)
            ax.fill_between(group["n"], group["ci_low"], group["ci_high"], alpha=0.15)
        _style(
            ax,
            labels.title_variant.format(variant=variant, fold=outer_fold),
            subset["ci_low"].to_numpy(dtype=float),
            subset["ci_high"].to_numpy(dtype=float),
        )
        ax.legend(ncol=4, fontsize=8, loc="best", title=labels.legend_tasks)
        fig.tight_layout()
        paths.extend(_save_figure(fig, out_dir, f"auc_vs_n_{variant}{labels.file_suffix}"))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for variant, group in mean_curves.groupby("variant", sort=False):
        group = group.sort_values("n")
        ax.plot(group["n"], group["roc_auc"], marker="o", markersize=3, label=variant)
        ax.fill_between(group["n"], group["ci_low"], group["ci_high"], alpha=0.15)
    _style(
        ax,
        labels.title_mean.format(fold=outer_fold),
        mean_curves["ci_low"].to_numpy(dtype=float),
        mean_curves["ci_high"].to_numpy(dtype=float),
    )
    ax.legend(fontsize=9, loc="best", title=labels.legend_variants)
    fig.tight_layout()
    paths.extend(_save_figure(fig, out_dir, f"auc_vs_n_mean{labels.file_suffix}"))
    plt.close(fig)
    return paths


def _save_figure(fig: "Figure", out_dir: Path, stem: str) -> List[Path]:
    """
    Save figure in every format from ``FIGURE_FORMATS``.

    Parameters
    ----------
    fig : Figure
        Рисунок matplotlib.
    out_dir : Path
        Каталог результата.
    stem : str
        Имя файла без расширения.

    Returns
    -------
    List[Path]
        Пути сохранённых файлов.
    """
    saved: List[Path] = []
    for fmt in FIGURE_FORMATS:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=150, format=fmt)
        saved.append(path)
    return saved


def render_report(
    curves: pd.DataFrame,
    mean_curves: pd.DataFrame,
    best: pd.DataFrame,
    n_bootstrap: int,
    outer_fold: Union[int, str],
    png_names: Sequence[str],
) -> str:
    """
    Render markdown report.

    Parameters
    ----------
    curves : pd.DataFrame
        Таблица кривых по ячейкам.
    mean_curves : pd.DataFrame
        Средние по задачам кривые.
    best : pd.DataFrame
        Таблица :func:`best_n_by_train`.
    n_bootstrap : int
        Число реплик (для подписи).
    outer_fold : Union[int, str]
        Outer fold или подпись набора fold.
    png_names : Sequence[str]
        Имена PNG для ссылок.

    Returns
    -------
    str
        Markdown-отчёт.
    """
    variants = list(dict.fromkeys(mean_curves["variant"]))
    ns = sorted(mean_curves["n"].unique())
    n_edges = max(ns)
    lines = [
        f"# Классификатор «средний вес top-n рёбер», test outer {outer_fold}",
        "",
        f"Скор — среднее знаковых весов n рёбер с наибольшим |w|; знак по train pool своего fold. "
        f"ROC-AUC [95% CI]: один кластерный bootstrap по субъектам всех указанных fold, B={n_bootstrap}.",
        "",
        *[f"![{name}]({name})" for name in png_names],
        "",
        "## Средний по задачам ROC-AUC",
        "",
        "| n | " + " | ".join(variants) + " |",
        "|---|" + "---|" * len(variants),
    ]
    for n in ns:
        cells = []
        for variant in variants:
            row = mean_curves[(mean_curves["variant"] == variant) & (mean_curves["n"] == n)].iloc[0]
            cells.append(f"{row['roc_auc']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]")
        label = "all" if n == n_edges else str(n)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Лучший n по train AUC и test AUC при нём",
        "",
        "n выбран по train (test не участвует). Вырожденные скоры исключены.",
        "",
        "| task | " + " | ".join(variants) + " |",
        "|---|" + "---|" * len(variants),
    ]
    for task in list(dict.fromkeys(best["task"])):
        cells = []
        for variant in variants:
            sel = best[(best["variant"] == variant) & (best["task"] == task)]
            if sel.empty:
                cells.append("—")
                continue
            row = sel.iloc[0]
            label = "all" if int(row["best_n"]) == n_edges else str(int(row["best_n"]))
            cells.append(
                f"n={label}: {row['roc_auc']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
            )
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def run_top_n(
    data_root: Path,
    out_dir: Path,
    tasks: Sequence[str],
    n_values: Sequence[int] = DEFAULT_N_VALUES,
    outer_folds: Sequence[int] = OUTER_FOLDS,
    n_bootstrap: int = 2000,
    seed: int = 42,
    workers: int = 8,
    use_cache: bool = True,
    plot_lang: str = DEFAULT_LANG,
) -> int:
    """
    Run top-n edge baseline.

    Parameters
    ----------
    data_root : Path
        Корень архива HCP.
    out_dir : Path
        Каталог результатов.
    tasks : Sequence[str]
        Задачи HCP.
    n_values : Sequence[int], optional
        Значения n (``все рёбра`` добавляется автоматически).
    outer_folds : Sequence[int], optional
        Outer fold; test всех fold объединяются, CI — по всем fold.
    n_bootstrap : int, optional
        Число bootstrap-реплик.
    seed : int, optional
        Random seed.
    workers : int, optional
        Потоки чтения матриц.
    use_cache : bool, optional
        Кэшировать скоры в ``out_dir/cache``.
    plot_lang : str, optional
        Язык подписей рисунков: ``ru``, ``en`` или ``both``.

    Returns
    -------
    int
        0 при успехе.
    """
    resolve_langs(plot_lang)
    unknown = sorted(set(tasks) - set(TASKS))
    if unknown:
        raise ValueError(f"Неизвестные задачи {unknown}, допустимы {TASKS}")
    if n_bootstrap < 100:
        raise ValueError(f"n_bootstrap слишком мал: {n_bootstrap}")

    folds = normalize_folds(outer_folds)
    root = Path(resolve_existing_path(str(data_root)))
    expected = expected_test_subjects(root / "splits.json", folds)
    # Число рёбер: верхний треугольник n_regions × n_regions (по первому графу).
    first_pool, _ = load_samples(root, VARIANTS[0], tasks[0], folds[0])
    n_regions = int(np.load(first_pool[0].edge_path).shape[0])
    ns = build_n_grid(n_regions * (n_regions - 1) // 2, n_values)
    log.info("Сетка n (%d значений): %s", len(ns), ns)

    cache_dir = out_dir / "cache" if use_cache else None
    tables: List[pd.DataFrame] = []
    mean_tables: List[pd.DataFrame] = []
    for variant in VARIANTS:
        comparisons: List[TaskComparison] = []
        for task in tasks:
            frames = [collect_scores(root, variant, task, fold, ns, workers, cache_dir) for fold in folds]
            validate_fold_subjects(
                {
                    fold: {"top_n": frame.loc[frame["role"] == "test", ["stem"]]}
                    for fold, frame in zip(folds, frames)
                },
                expected,
            )
            table, comparison = evaluate_cell_folds(frames, variant, task, ns, n_bootstrap, seed)
            tables.append(table)
            comparisons.append(comparison)
            top = table.iloc[table["train_auc"].fillna(0.0).argmax()]
            log.info(
                "%s/%s: лучший по train n=%d, test AUC %.4f",
                variant, task, int(top["n"]), top["roc_auc"],
            )
        if len(comparisons) > 1:
            mean_tables.append(mean_curve(variant, comparisons))
    return _write_outputs(
        pd.concat(tables, ignore_index=True),
        pd.concat(mean_tables, ignore_index=True) if mean_tables else pd.DataFrame(),
        out_dir,
        n_bootstrap,
        fold_label(folds),
        plot_lang,
    )


def _write_outputs(
    curves: pd.DataFrame,
    mean_curves: pd.DataFrame,
    out_dir: Path,
    n_bootstrap: int,
    outer_fold: Union[int, str],
    plot_lang: str = DEFAULT_LANG,
) -> int:
    langs = resolve_langs(plot_lang)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = best_n_by_train(curves)
    curves.to_csv(out_dir / "auc_by_n.csv", index=False)
    mean_curves.to_csv(out_dir / "auc_by_n_mean.csv", index=False)
    best.to_csv(out_dir / "best_n_by_train.csv", index=False)
    if mean_curves.empty:
        log.warning("Одна задача: сводный график и средняя таблица не строятся")
        return 0
    paths_by_lang = {lang: plot_curves(curves, mean_curves, out_dir, outer_fold, lang) for lang in langs}
    if DEFAULT_LANG not in paths_by_lang:
        log.info("Отчёт report.md не обновлён: он ссылается на рисунки на языке %s", DEFAULT_LANG)
    else:
        png_names = [p.name for p in paths_by_lang[DEFAULT_LANG] if p.suffix == ".png"]
        report = render_report(curves, mean_curves, best, n_bootstrap, outer_fold, png_names)
        (out_dir / "report.md").write_text(report, encoding="utf-8")
        print(report)
    log.info("Результаты записаны в %s", out_dir)
    return 0


def replot(
    out_dir: Path,
    outer_fold: Union[int, str],
    n_bootstrap: int,
    plot_lang: str = DEFAULT_LANG,
) -> int:
    """
    Replot from saved tables.

    Parameters
    ----------
    out_dir : Path
        Каталог с ``auc_by_n.csv`` и ``auc_by_n_mean.csv``.
    outer_fold : Union[int, str]
        Outer fold или подпись набора fold (для заголовков).
    n_bootstrap : int
        Число реплик (для подписи в отчёте).
    plot_lang : str, optional
        Язык подписей рисунков: ``ru``, ``en`` или ``both``.

    Returns
    -------
    int
        0 при успехе.
    """
    curves = pd.read_csv(out_dir / "auc_by_n.csv")
    mean_curves = pd.read_csv(out_dir / "auc_by_n_mean.csv")
    return _write_outputs(curves, mean_curves, out_dir, n_bootstrap, outer_fold, plot_lang)


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
    parser = argparse.ArgumentParser(
        description="Классификатор «средний вес top-n рёбер»: ROC-AUC(n) для каждого варианта",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / ".." / "HPC_data_ref" / "hcp_mmp_mean_ablation_graphs",
    )
    parser.add_argument("--task", action="append", default=None, help="Повторяемый; по умолчанию все 7")
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_N_VALUES),
        help="Сетка n; 'все рёбра' добавляется автоматически",
    )
    parser.add_argument(
        "--outer-fold",
        type=int,
        nargs="+",
        default=list(OUTER_FOLDS),
        metavar="K",
        help="Outer fold; test всех указанных fold объединяются в один отчёт (по умолчанию 0 1 2 3)",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-cache", action="store_true", help="Пересчитать скоры, не читать кэш")
    parser.add_argument(
        "--plot-lang",
        choices=PLOT_LANG_CHOICES,
        default=DEFAULT_LANG,
        help="Язык подписей рисунков (en — файлы с суффиксом _en; both — оба набора)",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Только перестроить графики и отчёт из сохранённых CSV",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "top_n_edge_baseline",
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)
    out_dir = args.out_dir.resolve()
    if args.replot:
        return replot(out_dir, fold_label(normalize_folds(args.outer_fold)), args.n_bootstrap, args.plot_lang)
    return run_top_n(
        data_root=args.data_root.resolve(),
        out_dir=out_dir,
        tasks=tuple(args.task) if args.task else TASKS,
        n_values=args.n_values,
        outer_folds=args.outer_fold,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        workers=args.workers,
        use_cache=not args.no_cache,
        plot_lang=args.plot_lang,
    )


if __name__ == "__main__":
    raise SystemExit(main())
