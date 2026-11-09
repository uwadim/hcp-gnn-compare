"""Тесты построения графов и индекса HCP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dataset import (
    HCPDataModule,
    adjacency_to_edge_tensors,
    label_to_binary,
)


def test_label_to_binary() -> None:
    """Метки 1/2 → 0/1."""
    assert label_to_binary(1) == 0.0
    assert label_to_binary(2) == 1.0
    with pytest.raises(ValueError):
        label_to_binary(3)


def test_adjacency_to_edge_tensors_symmetric() -> None:
    """Верхний треугольник → оба направления, одинаковые веса."""
    adj = np.array(
        [
            [0.0, 0.5, 0.0],
            [0.5, 0.0, -1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    edge_index, edge_attr = adjacency_to_edge_tensors(adj, drop_percentile=0.0)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == edge_attr.shape[0]
    assert edge_attr.shape[1] == 1
    # 2 undirected edges → 4 directed
    assert edge_index.shape[1] == 4
    # Симметрия: число (i,j) = число (j,i)
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    pairs = sorted(zip(src, dst))
    rev = sorted(zip(dst, src))
    assert pairs == rev


def test_adjacency_drop_percentile() -> None:
    """Отсечение слабых рёбер."""
    adj = np.zeros((4, 4), dtype=np.float64)
    adj[0, 1] = adj[1, 0] = 10.0
    adj[0, 2] = adj[2, 0] = 0.1
    adj[1, 2] = adj[2, 1] = 0.2
    edge_index, _ = adjacency_to_edge_tensors(adj, drop_percentile=50.0)
    # Остаётся хотя бы одно неориентированное ребро (2 directed).
    assert edge_index.shape[1] >= 2
    assert edge_index.shape[1] % 2 == 0


def test_datamodule_correlation_ones(data_root: Path) -> None:
    """Single-split correlation с node_features=ones (без pickle)."""
    cfg = OmegaConf.create(
        {
            "root": str(data_root),
            "variant": "correlation",
            "task": "WM",
            "node_features": "ones",
            "drop_percentile": 90.0,
            "edge_scaler": "none",
            "batch_size": 2,
            "num_workers": 0,
            "outer_fold": 0,
            "inner_val_fold": 0,
        }
    )
    dm = HCPDataModule(cfg)
    dm.setup_single_split()
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.test_dataset is not None
    assert len(dm.train_dataset) > 0
    assert len(dm.val_dataset) > 0
    assert len(dm.test_dataset) == 416  # 208 subjects × 2 states

    data = dm.train_dataset[0]
    assert data.x.shape == (379, 1)
    assert data.edge_index.shape[0] == 2
    assert data.edge_attr.ndim == 2
    assert data.y.numel() == 1

    batch = next(iter(dm.train_loader()))
    assert batch.x.size(0) % 379 == 0


def test_datamodule_ensemble_full(data_root: Path) -> None:
    """Ensemble full: train/val/test без пересечения субъектов."""
    cfg = OmegaConf.create(
        {
            "root": str(data_root),
            "variant": "ensemble_full",
            "task": "WM",
            "node_features": "ones",
            "drop_percentile": 90.0,
            "edge_scaler": "none",
            "batch_size": 2,
            "num_workers": 0,
            "outer_fold": 0,
            "inner_val_fold": 0,
        }
    )
    dm = HCPDataModule(cfg)
    dm.setup_single_split()
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    train_subj = set(dm.train_dataset.subjects)
    val_subj = set(dm.val_dataset.subjects)
    test_subj = set(dm.test_dataset.subjects) if dm.test_dataset else set()
    assert train_subj.isdisjoint(val_subj)
    assert train_subj.isdisjoint(test_subj)
    assert val_subj.isdisjoint(test_subj)
    # 624 train subjects × 2 states = 1248 в pool; val ≈ 208×2
    assert len(dm.val_dataset) == 416
    assert len(dm.train_dataset) == 832  # 416 subjects × 2
