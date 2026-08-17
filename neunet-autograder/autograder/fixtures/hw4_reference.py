"""Reference HW4 implementations used for deterministic checks."""

from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_adjacency_matrix(edges: list[tuple[int, int]], n_nodes: int) -> np.ndarray:
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return adj


def gcn_forward(
    x: np.ndarray,
    adj: np.ndarray,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
) -> np.ndarray:
    h = x
    for w, b in zip(weights, biases):
        h = adj @ h @ w + b
        h = relu(h)
    return sigmoid(h)


def neighborhood_sample(
    adj: np.ndarray,
    target: int,
    n_sample: int,
    seed: int = 0,
) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    layers = {"output": [target]}
    seen = {target}
    frontier = [target]

    for layer_name in ("hidden2", "hidden1", "input"):
        neighbors: list[int] = []
        for node in frontier:
            nbrs = np.where(adj[node] > 0)[0].tolist()
            neighbors.extend(nbr for nbr in nbrs if nbr not in seen)
        neighbors = list(dict.fromkeys(neighbors))
        if not neighbors:
            layers[layer_name] = []
            frontier = []
            continue
        k = min(n_sample, len(neighbors))
        chosen = rng.choice(neighbors, size=k, replace=False).tolist()
        layers[layer_name] = chosen
        seen.update(chosen)
        frontier = chosen

    return layers


def softmask(scores: np.ndarray, adj: np.ndarray) -> np.ndarray:
    masked = np.where(adj > 0, scores, -1e9)
    exp = np.exp(masked - masked.max(axis=-1, keepdims=True))
    denom = exp.sum(axis=-1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return exp / denom


def gat_forward(
    x: np.ndarray,
    adj: np.ndarray,
    weight: np.ndarray,
    attention: np.ndarray,
) -> np.ndarray:
    h = x @ weight
    n_nodes = h.shape[0]
    scores = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(n_nodes):
            pair = np.concatenate([h[i], h[j]])
            scores[i, j] = pair @ attention
    attn = softmask(scores, adj + np.eye(n_nodes))
    return attn @ h


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)]
N_NODES = 4
ADJ = build_adjacency_matrix(EDGES, N_NODES)
X = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
    ],
    dtype=np.float64,
)
W1 = np.array([[0.2, -0.1], [0.3, 0.4]], dtype=np.float64)
W2 = np.array([[0.5, -0.2], [0.1, 0.6]], dtype=np.float64)
W3 = np.array([[0.7, 0.3], [-0.4, 0.2]], dtype=np.float64)
B1 = np.array([0.1, -0.1], dtype=np.float64)
B2 = np.array([0.0, 0.2], dtype=np.float64)
B3 = np.array([-0.1, 0.05], dtype=np.float64)
GAT_W = np.array([[0.3, -0.2], [0.4, 0.1]], dtype=np.float64)
GAT_A = np.array([0.25, -0.25, 0.5, -0.5], dtype=np.float64)

REF_GCN = gcn_forward(X, ADJ, [W1, W2, W3], [B1, B2, B3])
REF_SAMPLE = neighborhood_sample(ADJ, target=1, n_sample=2, seed=42)
REF_GAT = gat_forward(X, ADJ, GAT_W, GAT_A)
