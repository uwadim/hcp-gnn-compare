"""Smoke-тесты forward CF-MSR и SkipGCN."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data

from dataset import adjacency_to_edge_tensors
from model_factory import build_model
import numpy as np


def _toy_batch(n_nodes: int = 8, input_dim: int = 2, batch_size: int = 2) -> Batch:
    graphs = []
    rng = np.random.default_rng(0)
    for _ in range(batch_size):
        adj = rng.normal(size=(n_nodes, n_nodes))
        adj = (adj + adj.T) / 2.0
        np.fill_diagonal(adj, 0.0)
        edge_index, edge_attr = adjacency_to_edge_tensors(adj, drop_percentile=0.0)
        x = torch.randn(n_nodes, input_dim)
        graphs.append(
            Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([1.0]),
            )
        )
    return Batch.from_data_list(graphs)


def test_build_cfmsr_forward_input_dim2() -> None:
    """CFMSR: input_dim=2, логиты (B, 1)."""
    cfg = OmegaConf.create(
        {
            "model_name": "cfmsr",
            "hidden_dim": 16,
            "num_layers": 1,
            "num_heads": 2,
            "ffn_mult": 2.0,
            "num_seeds": 2,
            "dropout": 0.1,
            "classifier_dropout": 0.1,
            "drop_edge": 0.0,
            "node_noise_std": 0.0,
            "edge_noise_std": 0.0,
            "use_node_embedding": True,
            "edge_batchnorm": True,
            "edge_readout": "none",
        }
    )
    model = build_model(cfg, num_nodes=8, input_dim=2)
    batch = _toy_batch(n_nodes=8, input_dim=2, batch_size=2)
    logits = model(batch)
    assert logits.shape == (2, 1)


def test_build_skipgcn_forward_input_dim1() -> None:
    """SkipGCN: input_dim=1, логиты (B, 1)."""
    cfg = OmegaConf.create(
        {
            "model_name": "skipgcn",
            "skipgcn": {
                "hidden_channels": 8,
                "dropout": 0.1,
                "pooling_type": "global_max_pool",
            },
        }
    )
    model = build_model(cfg, num_nodes=8, input_dim=1)
    batch = _toy_batch(n_nodes=8, input_dim=1, batch_size=2)
    logits = model(batch)
    assert logits.shape == (2, 1)
