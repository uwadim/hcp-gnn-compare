#!/usr/bin/env python3
"""
Логистическая регрессия на характеристиках полносвязных графов.

Признаки графа (матрица весов ``N×N``, верхний треугольник без диагонали):

* ``block_means`` — средние веса рёбер внутри блоков и между блоками узлов
  (``B(B+1)/2`` признаков для ``B`` блоков);
* ``strength_summary`` — сводки узловых сил ``s``, ``s+``, ``s−``, ``|s|``
  (среднее, std, min, max, асимметрия, эксцесс) и энтропия распределения ``|s|``;
* ``strength_vector`` — вектор из ``N`` знаковых узловых сил.

Узловая сила ``s_i = Σ_j w_ij``; ``s+_i`` — сумма положительных весов, ``s−_i`` —
модуль суммы отрицательных, ``|s|_i = s+_i + s−_i``.

Модель: ``StandardScaler`` → ``LogisticRegression`` (L2); сила регуляризации ``C``
выбирается на train pool по ROC-AUC при ``GroupKFold`` по субъектам, оценка —
на outer test. Для 4 вариантов данных × 7 задач считаются ROC-AUC [95% CI]
(кластерный bootstrap по субъектам) и парные ΔAUC с поправками на
множественность — теми же функциями, что в ``compare_variants.py``.

Блоки узлов. В архиве HCP-MMP нет разметки на функциональные сети, есть только
названия областей и MNI-координаты (``metadata/mmp_regions.csv``). По умолчанию
кора каждого полушария делится k-means по MNI-координатам на ``--n-spatial``
пространственных блоков (без меток классов → без утечки), подкорка, мозжечок и
ствол образуют один блок. Для Yeo/Cole–Anticevic/Glasser-22 передайте
``--partition-csv`` с колонками ``mmp_id,network``.

Результат: ``artifacts/graph_features_logreg/`` — ``summary.md``, ``cells.csv``,
``partition.csv`` и каталог ``<feature_set>/`` с одним отчётом ``report.md`` по всем
outer fold (pooled и macro AUC с CI, AUC по fold) и таблицами ``auc_{pooled,macro}.csv``,
``pairwise_{pooled,macro}.csv``, ``auc_per_fold.csv``.
"""

from __future__ import annotations

# 1. Standard library
import argparse
import hashlib
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

# 2. External libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# 3. Local
from batch_runner import OUTER_FOLDS, TASKS, VARIANTS  # noqa: E402
from compare_variants import DEFAULT_REFERENCE, MEAN_TASK  # noqa: E402
from compare_variants_cv import (  # noqa: E402
    CvTaskComparison,
    compare_task_cv,
    expected_test_subjects,
    normalize_folds,
    validate_fold_subjects,
    write_cv_comparison,
)
from config_utils import resolve_existing_path  # noqa: E402
from dataset import GraphSample  # noqa: E402
from mean_edge_baseline import load_samples  # noqa: E402

log = logging.getLogger("graph_features_logreg")

MODEL_NAME = "graph_features_logreg"
FEATURE_GROUPS = ("block_means", "strength_summary", "strength_vector")
ALL_FEATURE_SET = "all"
DEFAULT_FEATURE_SETS = ("block_means", "strength_summary", "strength_vector", ALL_FEATURE_SET)
DEFAULT_N_SPATIAL = 6
DEFAULT_C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
STRENGTH_KINDS = ("s", "s_pos", "s_neg", "s_abs")
STRENGTH_STATS = ("mean", "std", "min", "max", "skew", "kurtosis")
ENTROPY_NAME = "s_abs_entropy"
SUBCORTEX_BLOCK = "SUB"
CORTEX_PATTERN = re.compile(r"^[LR]_")
# Нижняя граница std: ниже — распределение считается константой.
STD_EPS = 1e-12
META_COLUMNS = ("stem", "subject", "label", "role")


# --------------------------------------------------------------------------- #
# Разбиение узлов на блоки
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Partition:
    """
    Разбиение узлов графа на блоки.

    Attributes
    ----------
    labels : np.ndarray
        Номер блока каждого узла, форма ``(N,)``, значения ``0..B-1``.
    names : Tuple[str, ...]
        Имена блоков, длина ``B``.
    """

    labels: np.ndarray
    names: Tuple[str, ...]

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        if labels.ndim != 1 or labels.size == 0 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"labels должны быть непустым 1D целочисленным массивом, shape={labels.shape}")
        if labels.min() < 0 or labels.max() >= len(self.names):
            raise ValueError(f"labels вне диапазона 0..{len(self.names) - 1}")
        sizes = np.bincount(labels, minlength=len(self.names))
        small = [self.names[i] for i in np.flatnonzero(sizes < 2)]
        if small:
            raise ValueError(f"В блоках {small} меньше 2 узлов: внутриблочное среднее не определено")
        object.__setattr__(self, "labels", labels.astype(np.int64))

    @property
    def n_nodes(self) -> int:
        """Число узлов."""
        return int(self.labels.size)

    @property
    def n_blocks(self) -> int:
        """Число блоков."""
        return len(self.names)

    @property
    def tag(self) -> str:
        """Короткий хэш разбиения для имён кэша."""
        digest = hashlib.sha1(self.labels.tobytes() + "|".join(self.names).encode("utf-8"))
        return f"p{digest.hexdigest()[:8]}"

    def indicator(self) -> np.ndarray:
        """
        Build indicator matrix.

        Returns
        -------
        np.ndarray
            Матрица ``(N, B)`` float64: 1, если узел принадлежит блоку.
        """
        out = np.zeros((self.n_nodes, self.n_blocks), dtype=np.float64)
        out[np.arange(self.n_nodes), self.labels] = 1.0
        return out


def read_regions(data_root: Path) -> pd.DataFrame:
    """
    Read region table of the archive.

    Parameters
    ----------
    data_root : Path
        Корень архива HCP.

    Returns
    -------
    pd.DataFrame
        Таблица ``mmp_regions.csv``, отсортированная по ``mmp_id`` (1..N).

    Raises
    ------
    ValueError
        Если нет нужных колонок или ``mmp_id`` не равен ``1..N``.
    """
    path = Path(resolve_existing_path(str(data_root / "metadata" / "mmp_regions.csv")))
    regions = pd.read_csv(path).sort_values("mmp_id").reset_index(drop=True)
    required = {"mmp_id", "label", "mni_x_mm", "mni_y_mm", "mni_z_mm"}
    missing = sorted(required - set(regions.columns))
    if missing:
        raise ValueError(f"В {path} нет колонок {missing}")
    if not np.array_equal(regions["mmp_id"].to_numpy(), np.arange(1, len(regions) + 1)):
        raise ValueError(f"mmp_id в {path} должен быть 1..{len(regions)} без пропусков")
    return regions


def spatial_partition(regions: pd.DataFrame, n_clusters: int, seed: int = 0) -> Partition:
    """
    Build spatial partition: k-means of cortex per hemisphere + one subcortical block.

    Parameters
    ----------
    regions : pd.DataFrame
        Результат :func:`read_regions`.
    n_clusters : int
        Число пространственных блоков в каждом полушарии.
    seed : int, optional
        Random seed k-means.

    Returns
    -------
    Partition
        Блоки ``L1..Lk`` (спереди назад), ``R1..Rk`` и ``SUB``.

    Raises
    ------
    ValueError
        Если ``n_clusters`` < 1 или в полушарии недостаточно областей коры.
    """
    if n_clusters < 1:
        raise ValueError(f"n_clusters должен быть ≥ 1, получено {n_clusters}")
    is_cortex = regions["label"].astype(str).str.match(CORTEX_PATTERN).to_numpy()
    labels = np.full(len(regions), -1, dtype=np.int64)
    names: List[str] = []
    for hemisphere in ("L", "R"):
        mask = is_cortex & regions["label"].astype(str).str.startswith(f"{hemisphere}_").to_numpy()
        if mask.sum() < 2 * n_clusters:
            raise ValueError(f"В полушарии {hemisphere} {int(mask.sum())} областей, мало для {n_clusters} блоков")
        coords = regions.loc[mask, ["mni_x_mm", "mni_y_mm", "mni_z_mm"]].to_numpy(dtype=np.float64)
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(coords)
        # Нумерация спереди назад: устойчивые имена блоков.
        rank = np.empty(n_clusters, dtype=np.int64)
        rank[np.argsort(-km.cluster_centers_[:, 1])] = np.arange(n_clusters)
        labels[mask] = len(names) + rank[km.labels_]
        names.extend(f"{hemisphere}{k + 1}" for k in range(n_clusters))
    if not (labels < 0).any():
        raise ValueError("Не найдено подкорковых областей: все области названы как L_*/R_*")
    labels[labels < 0] = len(names)
    names.append(SUBCORTEX_BLOCK)
    return Partition(labels=labels, names=tuple(names))


def load_partition_csv(path: Path, regions: pd.DataFrame) -> Partition:
    """
    Load partition from a CSV file.

    Parameters
    ----------
    path : Path
        CSV с колонками ``mmp_id`` и ``network``.
    regions : pd.DataFrame
        Результат :func:`read_regions`.

    Returns
    -------
    Partition
        Блоки в порядке первого появления в файле.

    Raises
    ------
    ValueError
        Если не хватает колонок, есть дубликаты ``mmp_id`` или не все области размечены.
    """
    frame = pd.read_csv(Path(resolve_existing_path(str(path))))
    missing = sorted({"mmp_id", "network"} - set(frame.columns))
    if missing:
        raise ValueError(f"В {path} нет колонок {missing}")
    if frame["mmp_id"].duplicated().any():
        raise ValueError(f"В {path} повторяются mmp_id")
    network = regions[["mmp_id"]].merge(frame[["mmp_id", "network"]], on="mmp_id", how="left")["network"]
    if network.isna().any():
        raise ValueError(f"В {path} не размечены области mmp_id={regions.loc[network.isna(), 'mmp_id'].tolist()}")
    names = tuple(dict.fromkeys(network.astype(str)))
    index = {name: i for i, name in enumerate(names)}
    return Partition(labels=np.array([index[str(v)] for v in network], dtype=np.int64), names=names)


# --------------------------------------------------------------------------- #
# Извлечение признаков
# --------------------------------------------------------------------------- #
def _moments(values: np.ndarray) -> Tuple[float, float, float, float]:
    """Среднее, std, асимметрия, эксцесс (избыточный); для константы асимметрия и эксцесс равны 0."""
    mean = float(values.mean())
    centered = values - mean
    std = float(np.sqrt(np.mean(centered**2)))
    if std < STD_EPS:
        return mean, 0.0, 0.0, 0.0
    skew = float(np.mean(centered**3) / std**3)
    kurtosis = float(np.mean(centered**4) / std**4 - 3.0)
    return mean, std, skew, kurtosis


def node_strengths(matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute node strengths.

    Parameters
    ----------
    matrix : np.ndarray
        Квадратная матрица весов ``(N, N)``; диагональ игнорируется.

    Returns
    -------
    Dict[str, np.ndarray]
        ``s`` (знаковая сумма), ``s_pos``, ``s_neg`` (модуль суммы отрицательных), ``s_abs``.
    """
    weights = np.array(matrix, dtype=np.float64)
    np.fill_diagonal(weights, 0.0)
    s_pos = np.clip(weights, 0.0, None).sum(axis=1)
    s_neg = -np.clip(weights, None, 0.0).sum(axis=1)
    return {"s": s_pos - s_neg, "s_pos": s_pos, "s_neg": s_neg, "s_abs": s_pos + s_neg}


def strength_summary(strengths: Mapping[str, np.ndarray]) -> np.ndarray:
    """
    Summarize node strengths.

    Parameters
    ----------
    strengths : Mapping[str, np.ndarray]
        Результат :func:`node_strengths`.

    Returns
    -------
    np.ndarray
        Вектор ``len(STRENGTH_KINDS) * len(STRENGTH_STATS) + 1``: сводки каждой силы
        и нормированная энтропия распределения ``|s|`` по узлам (1 — равномерное).
    """
    parts: List[float] = []
    for kind in STRENGTH_KINDS:
        values = strengths[kind]
        mean, std, skew, kurtosis = _moments(values)
        parts.extend((mean, std, float(values.min()), float(values.max()), skew, kurtosis))
    s_abs = strengths["s_abs"]
    total = float(s_abs.sum())
    if total > 0.0 and s_abs.size > 1:
        p = s_abs / total
        p = p[p > 0.0]
        entropy = float(-(p * np.log(p)).sum() / np.log(s_abs.size))
    else:
        entropy = 0.0
    parts.append(entropy)
    return np.asarray(parts, dtype=np.float64)


def strength_summary_names() -> List[str]:
    """Имена признаков :func:`strength_summary`."""
    return [f"{kind}_{stat}" for kind in STRENGTH_KINDS for stat in STRENGTH_STATS] + [ENTROPY_NAME]


class FeatureExtractor:
    """Извлекает признаки ``block_means``, ``strength_summary``, ``strength_vector`` из матрицы."""

    def __init__(self, partition: Partition) -> None:
        """
        Initialize extractor.

        Parameters
        ----------
        partition : Partition
            Разбиение узлов на блоки.
        """
        self._partition = partition
        self._indicator = partition.indicator()
        sizes = self._indicator.sum(axis=0)
        # Число упорядоченных пар узлов: внутри блока n(n-1), между блоками n_a·n_b.
        pairs = np.outer(sizes, sizes)
        pairs[np.diag_indices_from(pairs)] = sizes * (sizes - 1.0)
        self._pairs = pairs
        self._block_rows, self._block_cols = np.triu_indices(partition.n_blocks, k=0)

    @property
    def partition(self) -> Partition:
        """Разбиение узлов."""
        return self._partition

    def feature_names(self) -> Dict[str, List[str]]:
        """
        Build feature names per group.

        Returns
        -------
        Dict[str, List[str]]
            Имена признаков для каждой группы из ``FEATURE_GROUPS``.
        """
        names = self._partition.names
        return {
            "block_means": [f"{names[a]}-{names[b]}" for a, b in zip(self._block_rows, self._block_cols)],
            "strength_summary": strength_summary_names(),
            "strength_vector": [f"s_node{i + 1}" for i in range(self._partition.n_nodes)],
        }

    def extract(self, matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract features of one graph.

        Parameters
        ----------
        matrix : np.ndarray
            Симметричная матрица весов ``(N, N)``.

        Returns
        -------
        Dict[str, np.ndarray]
            Векторы признаков по группам из ``FEATURE_GROUPS``.

        Raises
        ------
        ValueError
            Если матрица не квадратная, не совпадает с разбиением по размеру или содержит NaN/Inf.
        """
        n = self._partition.n_nodes
        if matrix.ndim != 2 or matrix.shape != (n, n):
            raise ValueError(f"Ожидалась матрица {(n, n)}, получено {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError("Матрица весов содержит NaN или Inf")
        weights = np.array(matrix, dtype=np.float64)
        np.fill_diagonal(weights, 0.0)
        block_sums = self._indicator.T @ weights @ self._indicator
        block_means = block_sums / self._pairs
        strengths = node_strengths(weights)
        return {
            "block_means": block_means[self._block_rows, self._block_cols],
            "strength_summary": strength_summary(strengths),
            "strength_vector": strengths["s"],
        }


@dataclass(frozen=True)
class FeatureTable:
    """
    Признаки всех образцов ячейки variant × task.

    Attributes
    ----------
    meta : pd.DataFrame
        Колонки ``stem, subject, label, role`` (role — train|test).
    groups : Dict[str, np.ndarray]
        Матрицы признаков ``(n_samples, n_features)`` по группам.
    """

    meta: pd.DataFrame
    groups: Dict[str, np.ndarray]

    def select(self, groups: Sequence[str]) -> np.ndarray:
        """
        Concatenate feature groups.

        Parameters
        ----------
        groups : Sequence[str]
            Имена групп.

        Returns
        -------
        np.ndarray
            Матрица признаков ``(n_samples, Σ n_features)``.
        """
        return np.hstack([self.groups[g] for g in groups])


def parse_feature_set(spec: str) -> Tuple[str, ...]:
    """
    Parse feature set specification.

    Parameters
    ----------
    spec : str
        Группы через ``+`` (например ``block_means+strength_summary``) или ``all``.

    Returns
    -------
    Tuple[str, ...]
        Группы признаков без повторов в порядке ``FEATURE_GROUPS``.

    Raises
    ------
    ValueError
        Если указана неизвестная группа.
    """
    parts = list(FEATURE_GROUPS) if spec == ALL_FEATURE_SET else [p.strip() for p in spec.split("+")]
    unknown = sorted(set(parts) - set(FEATURE_GROUPS))
    if unknown or not parts:
        raise ValueError(f"Неизвестные группы признаков {unknown} в {spec!r}, допустимы {FEATURE_GROUPS}")
    return tuple(g for g in FEATURE_GROUPS if g in parts)


def feature_set_slug(spec: str) -> str:
    """Имя каталога для набора признаков."""
    return "_plus_".join(parse_feature_set(spec)) if spec != ALL_FEATURE_SET else ALL_FEATURE_SET


def _extract_file(path: Path, extractor: FeatureExtractor) -> Dict[str, np.ndarray]:
    return extractor.extract(np.load(path))


def build_table(
    pool_samples: Sequence[GraphSample],
    test_samples: Sequence[GraphSample],
    extractor: FeatureExtractor,
    workers: int,
) -> FeatureTable:
    """
    Build feature table of train pool and test.

    Parameters
    ----------
    pool_samples : Sequence[GraphSample]
        Образцы train pool.
    test_samples : Sequence[GraphSample]
        Образцы outer test.
    extractor : FeatureExtractor
        Извлекатель признаков.
    workers : int
        Число потоков чтения.

    Returns
    -------
    FeatureTable
        Таблица признаков; сначала train, затем test.
    """
    if workers < 1:
        raise ValueError(f"workers должен быть ≥ 1, получено {workers}")
    samples = [*pool_samples, *test_samples]
    if not pool_samples or not test_samples:
        raise ValueError("Пустой train pool или test")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda s: _extract_file(Path(s.edge_path), extractor), samples))
    meta = pd.DataFrame(
        {
            "stem": [str(s.stem) for s in samples],
            "subject": [str(s.subject) for s in samples],
            "label": [float(s.label) for s in samples],
            "role": ["train"] * len(pool_samples) + ["test"] * len(test_samples),
        }
    )
    groups = {g: np.vstack([row[g] for row in rows]) for g in FEATURE_GROUPS}
    return FeatureTable(meta=meta, groups=groups)


def _cache_path(cache_dir: Path, variant: str, task: str, outer_fold: int, tag: str) -> Path:
    return cache_dir / f"{variant}_{task}_outer{outer_fold}_{tag}.npz"


def save_table(table: FeatureTable, path: Path) -> None:
    """
    Save feature table to a compressed npz (без pickle).

    Parameters
    ----------
    table : FeatureTable
        Таблица признаков.
    path : Path
        Путь файла ``.npz``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Строки — как unicode-массивы: object-массивы потребовали бы pickle при чтении.
    arrays = {c: table.meta[c].to_numpy(dtype=str) for c in ("stem", "subject", "role")}
    arrays["label"] = table.meta["label"].to_numpy(dtype=np.float64)
    arrays.update({f"x_{g}": table.groups[g] for g in FEATURE_GROUPS})
    np.savez_compressed(path, **arrays)


def load_table(path: Path) -> Optional[FeatureTable]:
    """
    Load feature table from npz.

    Parameters
    ----------
    path : Path
        Путь файла ``.npz``.

    Returns
    -------
    Optional[FeatureTable]
        Таблица или ``None``, если файл отсутствует либо структура не подходит.
    """
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        needed = {*META_COLUMNS, *(f"x_{g}" for g in FEATURE_GROUPS)}
        if not needed <= set(data.files):
            log.warning("Кэш %s имеет другие поля — пересчёт", path)
            return None
        meta = pd.DataFrame({c: data[c] for c in META_COLUMNS})
        meta["label"] = meta["label"].astype(np.float64)
        for column in ("stem", "subject", "role"):
            meta[column] = meta[column].astype(str)
        return FeatureTable(meta=meta, groups={g: data[f"x_{g}"] for g in FEATURE_GROUPS})


def collect_features(
    data_root: Path,
    variant: str,
    task: str,
    outer_fold: int,
    extractor: FeatureExtractor,
    workers: int,
    cache_dir: Optional[Path],
) -> FeatureTable:
    """
    Collect (or load cached) features of one variant × task cell.

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
    extractor : FeatureExtractor
        Извлекатель признаков.
    workers : int
        Число потоков чтения.
    cache_dir : Optional[Path]
        Каталог кэша; ``None`` — без кэша.

    Returns
    -------
    FeatureTable
        Таблица признаков.
    """
    cache = None
    if cache_dir is not None:
        cache = _cache_path(cache_dir, variant, task, outer_fold, extractor.partition.tag)
        cached = load_table(cache)
        if cached is not None:
            return cached
    pool_samples, test_samples = load_samples(data_root, variant, task, outer_fold)
    table = build_table(pool_samples, test_samples, extractor, workers)
    if cache is not None:
        save_table(table, cache)
    return table


# --------------------------------------------------------------------------- #
# Модель и оценка
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LogRegConfig:
    """
    Настройки логистической регрессии.

    Attributes
    ----------
    c_grid : Tuple[float, ...]
        Сетка обратной силы L2-регуляризации ``C``.
    cv_folds : int
        Число фолдов ``GroupKFold`` для выбора ``C``.
    max_iter : int
        Предел итераций оптимизатора.
    """

    c_grid: Tuple[float, ...] = DEFAULT_C_GRID
    cv_folds: int = 5
    max_iter: int = 5000

    def __post_init__(self) -> None:
        if not self.c_grid or any(c <= 0 for c in self.c_grid):
            raise ValueError(f"c_grid должна содержать положительные значения, получено {self.c_grid}")
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds должен быть ≥ 2, получено {self.cv_folds}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter должен быть ≥ 1, получено {self.max_iter}")


@dataclass(frozen=True)
class CellResult:
    """
    Результат одной ячейки набор признаков × variant × task.

    Attributes
    ----------
    feature_set : str
        Спецификация набора признаков.
    variant : str
        Вариант данных.
    task : str
        Задача HCP.
    n_features : int
        Число признаков.
    best_c : float
        Выбранное ``C``.
    cv_auc : float
        Средний ROC-AUC на валидации при выбранном ``C``.
    test : pd.DataFrame
        Колонки ``stem``, ``prob``, ``label``.
    outer_fold : int
        Outer fold ячейки.
    """

    feature_set: str
    variant: str
    task: str
    n_features: int
    best_c: float
    cv_auc: float
    test: pd.DataFrame
    outer_fold: int = 0


def fit_predict_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: Sequence[str],
    x_test: np.ndarray,
    config: LogRegConfig,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, float, float]:
    """
    Fit logistic regression with grouped CV choice of C and predict test.

    Parameters
    ----------
    x_train : np.ndarray
        Признаки train, форма ``(n_train, F)``.
    y_train : np.ndarray
        Метки train (0/1).
    groups : Sequence[str]
        Субъект каждого образца train (для ``GroupKFold``).
    x_test : np.ndarray
        Признаки test, форма ``(n_test, F)``.
    config : LogRegConfig
        Настройки модели.
    n_jobs : int, optional
        Число процессов GridSearchCV.

    Returns
    -------
    Tuple[np.ndarray, float, float]
        Вероятности класса 1 на test, выбранное ``C`` и средний CV ROC-AUC.

    Raises
    ------
    ValueError
        Если размеры не согласованы, нет обоих классов или субъектов меньше фолдов.
    """
    y = np.asarray(y_train).astype(int)
    if x_train.shape[0] != y.size or y.size != len(groups):
        raise ValueError("x_train, y_train и groups должны иметь одинаковую длину")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError(f"Число признаков train {x_train.shape[1]} ≠ test {x_test.shape[1]}")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"В train должны быть оба класса, найдено {sorted(set(y.tolist()))}")
    if len(set(groups)) < config.cv_folds:
        raise ValueError(f"Субъектов {len(set(groups))} меньше cv_folds={config.cv_folds}")
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=config.max_iter)),
        ]
    )
    search = GridSearchCV(
        pipeline,
        param_grid={"clf__C": list(config.c_grid)},
        scoring="roc_auc",
        cv=GroupKFold(n_splits=config.cv_folds),
        refit=True,
        n_jobs=n_jobs,
    )
    search.fit(x_train, y, groups=np.asarray(groups))
    prob = search.predict_proba(x_test)[:, 1]
    return prob, float(search.best_params_["clf__C"]), float(search.best_score_)


def build_cell(
    table: FeatureTable,
    feature_set: str,
    variant: str,
    task: str,
    config: LogRegConfig,
    n_jobs: int = 1,
) -> CellResult:
    """
    Fit on train pool and predict outer test for one cell.

    Parameters
    ----------
    table : FeatureTable
        Признаки ячейки.
    feature_set : str
        Спецификация набора признаков.
    variant : str
        Вариант данных.
    task : str
        Задача HCP.
    config : LogRegConfig
        Настройки модели.
    n_jobs : int, optional
        Число процессов GridSearchCV.

    Returns
    -------
    CellResult
        Предсказания test и выбранное ``C``.
    """
    x_all = table.select(parse_feature_set(feature_set))
    is_train = (table.meta["role"] == "train").to_numpy()
    train, test = table.meta[is_train], table.meta[~is_train]
    prob, best_c, cv_auc = fit_predict_logreg(
        x_all[is_train],
        train["label"].to_numpy(),
        train["subject"].tolist(),
        x_all[~is_train],
        config,
        n_jobs,
    )
    return CellResult(
        feature_set=feature_set,
        variant=variant,
        task=task,
        n_features=int(x_all.shape[1]),
        best_c=best_c,
        cv_auc=cv_auc,
        test=pd.DataFrame(
            {"stem": test["stem"].to_numpy(), "prob": prob, "label": test["label"].to_numpy()}
        ),
    )


# --------------------------------------------------------------------------- #
# Отчёты
# --------------------------------------------------------------------------- #
def cells_table(cells: Sequence[CellResult]) -> pd.DataFrame:
    """
    Tabulate cells.

    Parameters
    ----------
    cells : Sequence[CellResult]
        Результаты ячеек.

    Returns
    -------
    pd.DataFrame
        ``fold, feature_set, variant, task, n_features, best_c, cv_auc``.
    """
    return pd.DataFrame(
        [
            {
                "fold": c.outer_fold,
                "feature_set": c.feature_set,
                "variant": c.variant,
                "task": c.task,
                "n_features": c.n_features,
                "best_c": c.best_c,
                "cv_auc": c.cv_auc,
            }
            for c in cells
        ]
    )


def render_summary(
    mean_aucs: pd.DataFrame,
    n_features: Mapping[str, int],
    partition: Partition,
    n_bootstrap: int,
    outer_fold: Union[int, str],
) -> str:
    """
    Render summary markdown across feature sets.

    Parameters
    ----------
    mean_aucs : pd.DataFrame
        Строки ``feature_set, variant, roc_auc, ci_low, ci_high`` (среднее по задачам).
    n_features : Mapping[str, int]
        Число признаков по наборам (по первой ячейке; не зависит от ячейки).
    partition : Partition
        Использованное разбиение узлов.
    n_bootstrap : int
        Число bootstrap-реплик.
    outer_fold : Union[int, str]
        Outer fold или подпись набора fold для заголовка.

    Returns
    -------
    str
        Markdown-отчёт.
    """
    variants = list(dict.fromkeys(mean_aucs["variant"]))
    sizes = np.bincount(partition.labels, minlength=partition.n_blocks)
    blocks = ", ".join(f"{name}={int(size)}" for name, size in zip(partition.names, sizes))
    lines = [
        f"# Логистическая регрессия на признаках графов, test outer {outer_fold}",
        "",
        "Средний по задачам ROC-AUC [95% CI] (pooled: test всех указанных fold вместе), "
        f"кластерный bootstrap по субъектам, B={n_bootstrap}. Модель: StandardScaler → LogisticRegression (L2), C по GroupKFold.",
        "",
        f"Блоки узлов ({partition.n_blocks}, размеры): {blocks}.",
        "",
        "| набор признаков | число признаков | " + " | ".join(variants) + " |",
        "|---|---|" + "---|" * len(variants),
    ]
    for feature_set in dict.fromkeys(mean_aucs["feature_set"]):
        sub = mean_aucs[mean_aucs["feature_set"] == feature_set].set_index("variant")
        cells = [
            f"{sub.loc[v, 'roc_auc']:.4f} [{sub.loc[v, 'ci_low']:.4f}, {sub.loc[v, 'ci_high']:.4f}]"
            for v in variants
        ]
        lines.append(f"| {feature_set} | {n_features[feature_set]} | " + " | ".join(cells) + " |")
    lines += ["", "Подробные таблицы по задачам — в `<набор признаков>/report.md`.", ""]
    return "\n".join(lines)


def partition_table(regions: pd.DataFrame, partition: Partition) -> pd.DataFrame:
    """
    Tabulate node-to-block assignment.

    Parameters
    ----------
    regions : pd.DataFrame
        Результат :func:`read_regions`.
    partition : Partition
        Разбиение узлов.

    Returns
    -------
    pd.DataFrame
        ``mmp_id, label, block``.
    """
    return pd.DataFrame(
        {
            "mmp_id": regions["mmp_id"],
            "label": regions["label"],
            "block": [partition.names[i] for i in partition.labels],
        }
    )


def run_graph_features(
    data_root: Path,
    out_dir: Path,
    tasks: Sequence[str],
    feature_sets: Sequence[str] = DEFAULT_FEATURE_SETS,
    n_spatial: int = DEFAULT_N_SPATIAL,
    partition_csv: Optional[Path] = None,
    outer_folds: Sequence[int] = OUTER_FOLDS,
    n_bootstrap: int = 10000,
    seed: int = 42,
    reference: Optional[str] = DEFAULT_REFERENCE,
    alpha: float = 0.05,
    workers: int = 8,
    use_cache: bool = True,
    config: Optional[LogRegConfig] = None,
    n_jobs: int = 1,
) -> int:
    """
    Run logistic regression on graph features for all variants and tasks.

    Parameters
    ----------
    data_root : Path
        Корень архива HCP.
    out_dir : Path
        Каталог результатов.
    tasks : Sequence[str]
        Задачи HCP.
    feature_sets : Sequence[str], optional
        Наборы признаков (группы через ``+`` или ``all``).
    n_spatial : int, optional
        Число пространственных блоков коры в полушарии (если нет ``partition_csv``).
    partition_csv : Optional[Path], optional
        CSV ``mmp_id,network`` с собственной разметкой узлов.
    outer_folds : Sequence[int], optional
        Outer fold; test всех fold объединяются, CI считаются по всем fold
        (pooled и macro), модель на каждом fold обучается на его train pool.
    n_bootstrap : int, optional
        Число bootstrap-реплик.
    seed : int, optional
        Random seed bootstrap и k-means.
    reference : Optional[str], optional
        Эталон основного семейства гипотез.
    alpha : float, optional
        Уровень значимости.
    workers : int, optional
        Потоки чтения матриц.
    use_cache : bool, optional
        Кэшировать признаки в ``out_dir/cache``.
    config : Optional[LogRegConfig], optional
        Настройки модели; по умолчанию :class:`LogRegConfig`.
    n_jobs : int, optional
        Процессы GridSearchCV.

    Returns
    -------
    int
        0 при успехе.
    """
    unknown = sorted(set(tasks) - set(TASKS))
    if unknown:
        raise ValueError(f"Неизвестные задачи {unknown}, допустимы {TASKS}")
    if n_bootstrap < 100:
        raise ValueError(f"n_bootstrap слишком мал: {n_bootstrap}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha должен быть в (0, 1), получено {alpha}")
    if not feature_sets:
        raise ValueError("feature_sets пуст")
    for spec in feature_sets:
        parse_feature_set(spec)
    folds = normalize_folds(outer_folds)
    config = config or LogRegConfig()

    root = Path(resolve_existing_path(str(data_root)))
    expected = expected_test_subjects(root / "splits.json", folds)
    regions = read_regions(root)
    partition = (
        load_partition_csv(partition_csv, regions)
        if partition_csv is not None
        else spatial_partition(regions, n_spatial, seed=seed)
    )
    extractor = FeatureExtractor(partition)
    log.info("Блоков узлов: %d (%s)", partition.n_blocks, ", ".join(partition.names))

    cache_dir = out_dir / "cache" if use_cache else None
    tables = {
        (variant, task, fold): collect_features(root, variant, task, fold, extractor, workers, cache_dir)
        for task in tasks
        for fold in folds
        for variant in VARIANTS
    }

    all_cells: List[CellResult] = []
    mean_rows: List[pd.DataFrame] = []
    n_features: Dict[str, int] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    folds_label = f"{', '.join(str(k) for k in folds)} (pooled)"
    for spec in feature_sets:
        results: List[CvTaskComparison] = []
        for task in tasks:
            preds_by_fold = {}
            for fold in folds:
                fold_cells = [
                    replace(
                        build_cell(tables[(variant, task, fold)], spec, variant, task, config, n_jobs),
                        outer_fold=fold,
                    )
                    for variant in VARIANTS
                ]
                for cell in fold_cells:
                    log.info(
                        "%s fold %d %s/%s: C=%g, CV AUC %.4f",
                        spec, fold, cell.variant, cell.task, cell.best_c, cell.cv_auc,
                    )
                all_cells.extend(fold_cells)
                n_features[spec] = fold_cells[0].n_features
                preds_by_fold[fold] = {c.variant: c.test for c in fold_cells}
            validate_fold_subjects(preds_by_fold, expected)
            results.append(compare_task_cv(task, preds_by_fold, n_bootstrap=n_bootstrap, seed=seed))
        set_dir = out_dir / feature_set_slug(spec)
        tables_cv, _ = write_cv_comparison(
            results,
            label=f"{MODEL_NAME}, {spec}",
            folds=folds,
            out_dir=set_dir,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            reference=reference,
            note=f"логистическая регрессия на признаках `{spec}`, обучена на train pool каждого fold",
            quiet=True,
        )
        aucs = tables_cv["pooled"][0]
        summary_task = MEAN_TASK if len(results) > 1 else tasks[0]
        mean_rows.append(aucs[aucs["task"] == summary_task].assign(feature_set=spec))

    cells_table(all_cells).to_csv(out_dir / "cells.csv", index=False)
    partition_table(regions, partition).to_csv(out_dir / "partition.csv", index=False)
    summary = render_summary(
        pd.concat(mean_rows, ignore_index=True), n_features, partition, n_bootstrap, folds_label
    )
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    log.info("Результаты записаны в %s", out_dir)
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
    parser = argparse.ArgumentParser(
        description="Логистическая регрессия на блочных средних и узловых силах графов",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / ".." / "HPC_data_ref" / "hcp_mmp_mean_ablation_graphs",
    )
    parser.add_argument("--task", action="append", default=None, help="Повторяемый; по умолчанию все 7")
    parser.add_argument(
        "--feature-set",
        action="append",
        default=None,
        help=f"Повторяемый; группы через '+' из {FEATURE_GROUPS} или 'all'; по умолчанию {DEFAULT_FEATURE_SETS}",
    )
    parser.add_argument("--n-spatial", type=int, default=DEFAULT_N_SPATIAL, help="Блоков коры в полушарии")
    parser.add_argument("--partition-csv", type=Path, default=None, help="CSV mmp_id,network")
    parser.add_argument(
        "--outer-fold",
        type=int,
        nargs="+",
        default=list(OUTER_FOLDS),
        metavar="K",
        help="Outer fold; test всех указанных fold объединяются в один отчёт (по умолчанию 0 1 2 3)",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE,
        help="Эталон основного семейства; 'none' — все пары в одном семействе",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=8, help="Потоки чтения матриц")
    parser.add_argument("--n-jobs", type=int, default=1, help="Процессы GridSearchCV")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--c-grid", type=float, nargs="+", default=list(DEFAULT_C_GRID))
    parser.add_argument("--no-cache", action="store_true", help="Пересчитать признаки, не читать кэш")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "graph_features_logreg",
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
    return run_graph_features(
        data_root=args.data_root.resolve(),
        out_dir=args.out_dir.resolve(),
        tasks=tuple(args.task) if args.task else TASKS,
        feature_sets=tuple(args.feature_set) if args.feature_set else DEFAULT_FEATURE_SETS,
        n_spatial=args.n_spatial,
        partition_csv=args.partition_csv,
        outer_folds=args.outer_fold,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        reference=None if args.reference.lower() == "none" else args.reference,
        alpha=args.alpha,
        workers=args.workers,
        use_cache=not args.no_cache,
        config=LogRegConfig(c_grid=tuple(args.c_grid), cv_folds=args.cv_folds),
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
