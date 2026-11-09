"""
HCP Dataset Module.

Загрузка correlation- и ensemble-графов из архива
``hcp_mmp_mean_ablation_graphs`` в объекты ``torch_geometric.data.Data``.
"""

from __future__ import annotations

# 1. Standard library
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 2. External libraries
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

# 3. Local
from config_utils import parse_optional_int, resolve_existing_path
from cv_splits import (
    OuterFoldSplit,
    load_splits,
    select_outer_folds,
    subjects_to_set,
    validate_outer_folds,
)

log = logging.getLogger(__name__)

SUPPORTED_VARIANTS = (
    "correlation",
    "ensemble_correlation",
    "ensemble_univariate",
    "ensemble_full",
)
SUPPORTED_NODE_FEATURES = ("mean_std", "ones")
SUPPORTED_SHUFFLE_SCOPES = ("all", "train")
ENSEMBLE_VARIANT_DIR = {
    "ensemble_correlation": "correlation",
    "ensemble_univariate": "univariate",
    "ensemble_full": "full",
}
N_REGIONS_DEFAULT = 379


def label_to_binary(label: Any) -> float:
    """
    Label to binary.

    Преобразует метку архива (1=state1, 2=state2) в бинарный таргет
    (0 / 1). Положительный класс — state2.

    Parameters
    ----------
    label : Any
        Исходная метка.

    Returns
    -------
    float
        ``1.0`` для state2, ``0.0`` для state1.
    """
    value = int(label)
    if value == 2:
        return 1.0
    if value == 1:
        return 0.0
    raise ValueError(f"Ожидалась метка 1 или 2, получено {label!r}")


def load_vertex_features(path: Path) -> np.ndarray:
    """
    Load vertex features.

    Читает pickle DataFrame с колонками ``mean``, ``std``.

    Parameters
    ----------
    path : Path
        Путь к ``*.pickle``.

    Returns
    -------
    np.ndarray
        Массив формы ``(N, 2)``, float32.
    """
    try:
        frame = pd.read_pickle(path)
    except Exception:
        import joblib

        frame = joblib.load(path)

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Ожидался DataFrame в {path}, получено {type(frame)}")
    missing = [col for col in ("mean", "std") if col not in frame.columns]
    if missing:
        raise ValueError(f"В {path} нет колонок {missing}")
    values = frame[["mean", "std"]].to_numpy(dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Неверная форма признаков узлов {values.shape} в {path}")
    return values


def adjacency_to_edge_tensors(
    adj: np.ndarray,
    drop_percentile: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adjacency to edge tensors.

    Симметричную матрицу переводит в неориентированные рёбра (оба направления).

    Parameters
    ----------
    adj : np.ndarray
        Квадратная матрица весов.
    drop_percentile : float, optional
        Перцентиль по ``|w|``: рёбра ниже порога отбрасываются.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``edge_index`` ``(2, E)`` и ``edge_attr`` ``(E, 1)``.
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Ожидалась квадратная матрица, получено {adj.shape}")

    n = int(adj.shape[0])
    # Строим неориентированный граф по верхнему треугольнику.
    rows, cols = np.triu_indices(n, k=1)
    weights = adj[rows, cols].astype(np.float64)

    # Нулевые веса не создают рёбер.
    nonzero = np.abs(weights) > 0.0
    rows = rows[nonzero]
    cols = cols[nonzero]
    weights = weights[nonzero]

    if weights.size == 0:
        empty_index = torch.zeros((2, 0), dtype=torch.long)
        empty_attr = torch.zeros((0, 1), dtype=torch.float32)
        return empty_index, empty_attr

    if drop_percentile > 0.0:
        abs_w = np.abs(weights)
        threshold = float(np.percentile(abs_w, float(drop_percentile)))
        keep = abs_w >= threshold
        if not keep.any():
            # Оставляем самое сильное ребро.
            keep[int(np.argmax(abs_w))] = True
        rows = rows[keep]
        cols = cols[keep]
        weights = weights[keep]

    src = np.concatenate([rows, cols])
    dst = np.concatenate([cols, rows])
    w = np.concatenate([weights, weights]).astype(np.float32)

    edge_index = torch.from_numpy(np.stack([src, dst], axis=0).astype(np.int64))
    edge_attr = torch.from_numpy(w).view(-1, 1)
    return edge_index, edge_attr


def scale_edge_attr(
    edge_attr: torch.Tensor,
    mean: float,
    std: float,
) -> torch.Tensor:
    """
    Scale edge attr.

    Parameters
    ----------
    edge_attr : torch.Tensor
        Признаки рёбер.
    mean : float
        Среднее train.
    std : float
        Std train.

    Returns
    -------
    torch.Tensor
        Нормированные признаки.
    """
    denom = std if std > 1e-8 else 1.0
    return (edge_attr - mean) / denom


@dataclass
class GraphSample:
    """
    Graph sample.

    Описание одного графа до материализации ``Data``.

    Attributes
    ----------
    stem : str
        Имя файла без расширения.
    subject : str
        Id субъекта.
    state : str
        Имя состояния задачи.
    label : float
        Бинарная метка 0/1.
    edge_path : Path
        Путь к матрице рёбер ``.npy``.
    vertex_path : Optional[Path]
        Путь к признакам узлов (только для mean_std).
    outer_fold : int
        Outer fold, к которому относится файл (для ensemble).
    inner_fold : Optional[int]
        Inner fold каталога train (None для test / correlation).
    split_role : str
        ``train_pool`` | ``test``.
    """

    stem: str
    subject: str
    state: str
    label: float
    edge_path: Path
    vertex_path: Optional[Path]
    outer_fold: int
    inner_fold: Optional[int]
    split_role: str


def shuffle_sample_labels(
    samples: Sequence[GraphSample],
    seed: int,
) -> None:
    """
    Shuffle sample labels.

    Переставляет метки на месте (пермутация существующего вектора меток).
    Разрушает связь граф ↔ метка при сохранении баланса классов.

    Parameters
    ----------
    samples : Sequence[GraphSample]
        Образцы, у которых меняется ``label``.
    seed : int
        Random seed.

    Raises
    ------
    ValueError
        Если список пуст.
    """
    if not samples:
        raise ValueError("Нельзя перемешать метки пустого списка samples")
    labels = [float(s.label) for s in samples]
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(labels))
    for sample, idx in zip(samples, perm):
        sample.label = labels[int(idx)]


class HCPGraphDataset(Dataset):
    """
    HCPGraphDataset.

    In-memory / on-demand загрузка графов по списку ``GraphSample``.
    """

    def __init__(
        self,
        samples: Sequence[GraphSample],
        node_features: str = "mean_std",
        drop_percentile: float = 0.0,
        edge_mean: float = 0.0,
        edge_std: float = 1.0,
        edge_scaler: str = "none",
        n_regions: int = N_REGIONS_DEFAULT,
    ) -> None:
        """
        Initialize dataset.

        Parameters
        ----------
        samples : Sequence[GraphSample]
            Список образцов.
        node_features : str, optional
            ``mean_std`` или ``ones``.
        drop_percentile : float, optional
            Перцентиль отсечения рёбер.
        edge_mean : float, optional
            Среднее для scaler.
        edge_std : float, optional
            Std для scaler.
        edge_scaler : str, optional
            ``none`` или ``global``.
        n_regions : int, optional
            Число узлов.
        """
        super().__init__()
        self.samples = list(samples)
        self.node_features = str(node_features)
        self.drop_percentile = float(drop_percentile)
        self.edge_mean = float(edge_mean)
        self.edge_std = float(edge_std)
        self.edge_scaler = str(edge_scaler)
        self.n_regions = int(n_regions)

        if self.node_features not in SUPPORTED_NODE_FEATURES:
            raise ValueError(
                f"node_features должен быть одним из {SUPPORTED_NODE_FEATURES}"
            )

        self.labels = np.asarray([s.label for s in self.samples], dtype=np.float32)
        self.sample_ids = [s.stem for s in self.samples]
        self.subjects = [s.subject for s in self.samples]

    def len(self) -> int:
        """
        Len.

        Returns
        -------
        int
            Число графов.
        """
        return len(self.samples)

    def get(self, idx: int) -> Data:
        """
        Get.

        Parameters
        ----------
        idx : int
            Индекс образца.

        Returns
        -------
        Data
            Граф PyG.
        """
        sample = self.samples[idx]
        adj = np.load(sample.edge_path)
        if adj.shape[0] != self.n_regions or adj.shape[1] != self.n_regions:
            raise ValueError(
                f"Ожидалась матрица {(self.n_regions, self.n_regions)}, "
                f"получено {adj.shape} в {sample.edge_path}"
            )

        edge_index, edge_attr = adjacency_to_edge_tensors(
            adj,
            drop_percentile=self.drop_percentile,
        )
        if self.edge_scaler == "global":
            edge_attr = scale_edge_attr(edge_attr, self.edge_mean, self.edge_std)

        if self.node_features == "ones":
            x = torch.ones((self.n_regions, 1), dtype=torch.float32)
        else:
            if sample.vertex_path is None:
                raise FileNotFoundError(
                    f"node_features=mean_std, но нет vertex_path для {sample.stem}"
                )
            feats = load_vertex_features(sample.vertex_path)
            if feats.shape[0] != self.n_regions:
                raise ValueError(
                    f"Число регионов {feats.shape[0]} != {self.n_regions} "
                    f"в {sample.vertex_path}"
                )
            x = torch.from_numpy(feats)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([sample.label], dtype=torch.float32),
        )
        data.sample_idx = torch.tensor([idx], dtype=torch.long)
        data.stem = sample.stem
        data.subject = sample.subject
        return data


def _read_sample_order(root: Path, task: str) -> pd.DataFrame:
    path = root / "metadata" / f"{task}_mean_sample_order.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Не найден sample_order: {path}")
    frame = pd.read_csv(path)
    required = {"sample_index", "subject", "state", "label", "stem"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"В {path} нет колонок {sorted(missing)}")
    frame = frame.copy()
    frame["subject"] = frame["subject"].astype(str)
    frame["stem"] = frame["stem"].astype(str)
    frame["state"] = frame["state"].astype(str)
    frame["y"] = frame["label"].map(label_to_binary)
    return frame


def _correlation_paths(root: Path, task: str, stem: str) -> Tuple[Path, Path]:
    edge = (
        root
        / "correlation"
        / "graphs"
        / "edges"
        / "mean"
        / "pearson_cross_correlation"
        / task
        / f"{stem}.npy"
    )
    vertex = (
        root
        / "correlation"
        / "graphs"
        / "vertices"
        / "mean"
        / task
        / f"{stem}.pickle"
    )
    return edge, vertex


def _ensemble_root(root: Path, variant: str) -> Path:
    dirname = ENSEMBLE_VARIANT_DIR[variant]
    return root / "ensemble" / dirname / "graphs" / "mean" / "ensemble_edges"


def _build_correlation_samples(
    root: Path,
    task: str,
    order: pd.DataFrame,
    node_features: str,
) -> List[GraphSample]:
    samples: List[GraphSample] = []
    for row in order.itertuples(index=False):
        edge_path, vertex_path = _correlation_paths(root, task, str(row.stem))
        if not edge_path.is_file():
            raise FileNotFoundError(f"Нет correlation edges: {edge_path}")
        v_path: Optional[Path] = vertex_path if node_features == "mean_std" else None
        if v_path is not None and not v_path.is_file():
            raise FileNotFoundError(f"Нет vertex features: {v_path}")
        samples.append(
            GraphSample(
                stem=str(row.stem),
                subject=str(row.subject),
                state=str(row.state),
                label=float(row.y),
                edge_path=edge_path,
                vertex_path=v_path,
                outer_fold=-1,
                inner_fold=None,
                split_role="train_pool",
            )
        )
    return samples


def _build_ensemble_samples_for_fold(
    root: Path,
    variant: str,
    task: str,
    order: pd.DataFrame,
    outer: OuterFoldSplit,
    node_features: str,
) -> List[GraphSample]:
    ens_root = _ensemble_root(root, variant)
    fold_dir = ens_root / f"fold{outer.fold}"
    if not fold_dir.is_dir():
        raise FileNotFoundError(f"Нет каталога ensemble fold: {fold_dir}")

    order_by_stem = {str(row.stem): row for row in order.itertuples(index=False)}
    samples: List[GraphSample] = []

    # Train pool: каждый субъект outer.train ровно в одном inner-каталоге.
    for inner in outer.inner:
        inner_id = int(inner["fold"])
        inner_dir = fold_dir / "train" / f"fold{inner_id}" / task
        if not inner_dir.is_dir():
            raise FileNotFoundError(f"Нет inner train каталога: {inner_dir}")
        for edge_path in sorted(inner_dir.glob("*.npy")):
            stem = edge_path.stem
            row = order_by_stem.get(stem)
            if row is None:
                raise KeyError(f"stem {stem} отсутствует в sample_order")
            v_path: Optional[Path] = None
            if node_features == "mean_std":
                _, v_path = _correlation_paths(root, task, stem)
                if not v_path.is_file():
                    raise FileNotFoundError(f"Нет vertex features: {v_path}")
            samples.append(
                GraphSample(
                    stem=stem,
                    subject=str(row.subject),
                    state=str(row.state),
                    label=float(row.y),
                    edge_path=edge_path,
                    vertex_path=v_path,
                    outer_fold=outer.fold,
                    inner_fold=inner_id,
                    split_role="train_pool",
                )
            )

    test_dir = fold_dir / "test" / task
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Нет test каталога: {test_dir}")
    test_subjects = subjects_to_set(outer.test_subjects)
    for edge_path in sorted(test_dir.glob("*.npy")):
        stem = edge_path.stem
        row = order_by_stem.get(stem)
        if row is None:
            raise KeyError(f"stem {stem} отсутствует в sample_order")
        if str(row.subject) not in test_subjects:
            continue
        v_path = None
        if node_features == "mean_std":
            _, v_path = _correlation_paths(root, task, stem)
            if not v_path.is_file():
                raise FileNotFoundError(f"Нет vertex features: {v_path}")
        samples.append(
            GraphSample(
                stem=stem,
                subject=str(row.subject),
                state=str(row.state),
                label=float(row.y),
                edge_path=edge_path,
                vertex_path=v_path,
                outer_fold=outer.fold,
                inner_fold=None,
                split_role="test",
            )
        )
    return samples


def _estimate_edge_stats(
    samples: Sequence[GraphSample],
    drop_percentile: float,
    max_graphs: int = 32,
) -> Tuple[float, float]:
    if not samples:
        return 0.0, 1.0
    rng = np.random.default_rng(0)
    idxs = np.arange(len(samples))
    if len(idxs) > max_graphs:
        idxs = rng.choice(idxs, size=max_graphs, replace=False)
    values: List[np.ndarray] = []
    for i in idxs:
        adj = np.load(samples[int(i)].edge_path)
        rows, cols = np.triu_indices(adj.shape[0], k=1)
        w = adj[rows, cols].astype(np.float64)
        if drop_percentile > 0.0:
            thr = float(np.percentile(np.abs(w), drop_percentile))
            w = w[np.abs(w) >= thr]
        if w.size:
            values.append(w)
    if not values:
        return 0.0, 1.0
    concat = np.concatenate(values)
    return float(concat.mean()), float(concat.std() + 1e-8)


class HCPDataModule:
    """
    HCPDataModule.

    Готовит loaders для одного outer fold или полного CV.
    """

    def __init__(
        self,
        data_cfg: DictConfig,
        training_cfg: Optional[DictConfig] = None,
    ) -> None:
        """
        Initialize data module.

        Parameters
        ----------
        data_cfg : DictConfig
            Ветка ``data``.
        training_cfg : Optional[DictConfig], optional
            Ветка ``training`` (не используется напрямую).
        """
        self.data_cfg = data_cfg
        self.training_cfg = training_cfg
        self.root = Path(resolve_existing_path(data_cfg.root))
        self.variant = str(data_cfg.get("variant", "correlation")).strip().lower()
        self.task = str(data_cfg.get("task", "WM")).strip().upper()
        self.node_features = str(
            data_cfg.get("node_features", "mean_std")
        ).strip().lower()
        self.drop_percentile = float(data_cfg.get("drop_percentile", 0.0) or 0.0)
        self.edge_scaler = str(data_cfg.get("edge_scaler", "none")).strip().lower()
        self.batch_size = int(data_cfg.get("batch_size", 4))
        self.num_workers = int(data_cfg.get("num_workers", 0))
        self.outer_fold = parse_optional_int(data_cfg.get("outer_fold", None))
        self.inner_val_fold = int(data_cfg.get("inner_val_fold", 0))
        self.shuffle_labels = bool(data_cfg.get("shuffle_labels", False))
        self.shuffle_labels_seed = int(data_cfg.get("shuffle_labels_seed", 0) or 0)
        self.shuffle_labels_scope = str(
            data_cfg.get("shuffle_labels_scope", "all")
        ).strip().lower()

        if self.variant not in SUPPORTED_VARIANTS:
            raise ValueError(
                f"data.variant должен быть одним из {SUPPORTED_VARIANTS}, "
                f"получено {self.variant!r}"
            )
        if self.node_features not in SUPPORTED_NODE_FEATURES:
            raise ValueError(
                f"data.node_features должен быть одним из {SUPPORTED_NODE_FEATURES}"
            )
        if self.drop_percentile < 0.0 or self.drop_percentile >= 100.0:
            raise ValueError(
                f"drop_percentile должен быть в [0, 100), получено {self.drop_percentile}"
            )
        if self.shuffle_labels_scope not in SUPPORTED_SHUFFLE_SCOPES:
            raise ValueError(
                f"data.shuffle_labels_scope должен быть одним из "
                f"{SUPPORTED_SHUFFLE_SCOPES}, получено {self.shuffle_labels_scope!r}"
            )

        project_path = self.root / "metadata" / "project.json"
        self.n_regions = N_REGIONS_DEFAULT
        if project_path.is_file():
            import json

            with project_path.open("r", encoding="utf-8") as handle:
                project = json.load(handle)
            self.n_regions = int(project.get("n_regions", N_REGIONS_DEFAULT))

        self.splits = load_splits(self.root / "splits.json")
        validate_outer_folds(self.splits)
        self.order = _read_sample_order(self.root, self.task)

        self.train_dataset: Optional[HCPGraphDataset] = None
        self.val_dataset: Optional[HCPGraphDataset] = None
        self.test_dataset: Optional[HCPGraphDataset] = None
        self.pos_weight: float = 1.0
        self.node_input_dim: int = 2 if self.node_features == "mean_std" else 1
        self.num_nodes: int = self.n_regions

        self._pool_samples: List[GraphSample] = []
        self._test_samples: List[GraphSample] = []
        self._active_outer: Optional[OuterFoldSplit] = None

    def setup_outer(self, outer: OuterFoldSplit) -> None:
        """
        Setup outer.

        Загружает пул train/test образцов для одного outer fold.

        Parameters
        ----------
        outer : OuterFoldSplit
            Outer fold.
        """
        self._active_outer = outer
        if self.variant == "correlation":
            all_samples = _build_correlation_samples(
                self.root,
                self.task,
                self.order,
                self.node_features,
            )
            train_set = subjects_to_set(outer.train_subjects)
            test_set = subjects_to_set(outer.test_subjects)
            self._pool_samples = [
                s for s in all_samples if s.subject in train_set
            ]
            self._test_samples = [
                s for s in all_samples if s.subject in test_set
            ]
            # Для correlation помечаем inner_fold по принадлежности субъекта.
            subject_to_inner: Dict[str, int] = {}
            for inner in outer.inner:
                for subject in inner["test"]:
                    subject_to_inner[str(subject)] = int(inner["fold"])
            for sample in self._pool_samples:
                sample.inner_fold = subject_to_inner.get(sample.subject)
        else:
            all_samples = _build_ensemble_samples_for_fold(
                self.root,
                self.variant,
                self.task,
                self.order,
                outer,
                self.node_features,
            )
            self._pool_samples = [
                s for s in all_samples if s.split_role == "train_pool"
            ]
            self._test_samples = [
                s for s in all_samples if s.split_role == "test"
            ]

        self._maybe_shuffle_labels()
        log.info(
            "Outer fold=%d: train_pool=%d test=%d variant=%s task=%s",
            outer.fold,
            len(self._pool_samples),
            len(self._test_samples),
            self.variant,
            self.task,
        )

    @property
    def pool_samples(self) -> List[GraphSample]:
        """
        Pool samples.

        Returns
        -------
        List[GraphSample]
            Train pool активного outer fold (после ``setup_outer``).
        """
        return list(self._pool_samples)

    @property
    def test_samples(self) -> List[GraphSample]:
        """
        Test samples.

        Returns
        -------
        List[GraphSample]
            Outer test активного outer fold (после ``setup_outer``).
        """
        return list(self._test_samples)

    def _maybe_shuffle_labels(self) -> None:
        """
        Maybe shuffle labels.

        При ``shuffle_labels=true`` переставляет метки: ``all`` — pool+test,
        ``train`` — только train pool (test остаётся истинным).
        """
        if not self.shuffle_labels:
            return
        if self.shuffle_labels_scope == "all":
            targets = [*self._pool_samples, *self._test_samples]
        else:
            targets = list(self._pool_samples)
        before = [s.label for s in targets]
        shuffle_sample_labels(targets, seed=self.shuffle_labels_seed)
        n_changed = sum(a != b for a, b in zip(before, (s.label for s in targets)))
        log.warning(
            "shuffle_labels=true scope=%s seed=%d: изменено %d/%d меток",
            self.shuffle_labels_scope,
            self.shuffle_labels_seed,
            n_changed,
            len(targets),
        )

    def prepare_inner(
        self,
        inner_val_fold: int,
    ) -> None:
        """
        Prepare inner.

        Делит train pool на train/val по inner fold; test — outer test.

        Parameters
        ----------
        inner_val_fold : int
            Номер inner fold, используемого как validation.
        """
        if self._active_outer is None:
            raise RuntimeError("Сначала вызовите setup_outer")

        val_samples = [
            s for s in self._pool_samples if s.inner_fold == int(inner_val_fold)
        ]
        train_samples = [
            s
            for s in self._pool_samples
            if s.inner_fold is not None and s.inner_fold != int(inner_val_fold)
        ]
        if not train_samples or not val_samples:
            raise ValueError(
                f"Пустой train/val для inner_val_fold={inner_val_fold}: "
                f"train={len(train_samples)} val={len(val_samples)}"
            )

        edge_mean, edge_std = 0.0, 1.0
        if self.edge_scaler == "global":
            edge_mean, edge_std = _estimate_edge_stats(
                train_samples,
                drop_percentile=self.drop_percentile,
            )

        self.train_dataset = HCPGraphDataset(
            train_samples,
            node_features=self.node_features,
            drop_percentile=self.drop_percentile,
            edge_mean=edge_mean,
            edge_std=edge_std,
            edge_scaler=self.edge_scaler,
            n_regions=self.n_regions,
        )
        self.val_dataset = HCPGraphDataset(
            val_samples,
            node_features=self.node_features,
            drop_percentile=self.drop_percentile,
            edge_mean=edge_mean,
            edge_std=edge_std,
            edge_scaler=self.edge_scaler,
            n_regions=self.n_regions,
        )
        self.test_dataset = HCPGraphDataset(
            self._test_samples,
            node_features=self.node_features,
            drop_percentile=self.drop_percentile,
            edge_mean=edge_mean,
            edge_std=edge_std,
            edge_scaler=self.edge_scaler,
            n_regions=self.n_regions,
        )
        self.node_input_dim = self.train_dataset[0].x.size(-1)
        self.num_nodes = int(self.train_dataset[0].x.size(0))
        self.pos_weight = self._compute_pos_weight(self.train_dataset.labels)

    def setup_single_split(self) -> None:
        """
        Setup single split.

        Один outer fold (``data.outer_fold`` или 0) и один inner val fold.
        """
        folds = select_outer_folds(self.splits, self.outer_fold)
        outer = folds[0]
        self.setup_outer(outer)
        self.prepare_inner(self.inner_val_fold)

    def _compute_pos_weight(self, labels: np.ndarray) -> float:
        y = np.asarray(labels, dtype=np.float32)
        n_pos = float((y >= 0.5).sum())
        n_neg = float((y < 0.5).sum())
        if n_pos <= 0.0:
            return 1.0
        return max(n_neg / n_pos, 1e-3)

    def _loader(self, dataset: Optional[HCPGraphDataset], shuffle: bool) -> Optional[DataLoader]:
        if dataset is None or len(dataset) == 0:
            return None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def train_loader(self) -> DataLoader:
        """
        Train loader.

        Returns
        -------
        DataLoader
            Загрузчик train.
        """
        loader = self._loader(self.train_dataset, shuffle=True)
        assert loader is not None
        return loader

    def val_loader(self) -> Optional[DataLoader]:
        """
        Val loader.

        Returns
        -------
        Optional[DataLoader]
            Загрузчик validation.
        """
        return self._loader(self.val_dataset, shuffle=False)

    def test_loader(self) -> Optional[DataLoader]:
        """
        Test loader.

        Returns
        -------
        Optional[DataLoader]
            Загрузчик outer test.
        """
        return self._loader(self.test_dataset, shuffle=False)
