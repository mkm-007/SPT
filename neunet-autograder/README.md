# NeuNet Autograder

Automated grading for **CSC 8851 Deep Learning** notebook submissions.

## Quick start

```bash
cd neunet-autograder
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
neunet-grade samples/CSC8851_S2026_HW4_Sample.ipynb --hw 4
```

## What it grades

| HW | Checks |
|---|---|
| **HW4** | Filename, execution, adjacency matrix, GCN, neighborhood sampling, GAT (numerical) |
| **HW1–3** | Filename, execution, required keywords in code (Tier 1) |

## CLI

```bash
neunet-grade path/to/notebook.ipynb --hw 4
neunet-grade path/to/notebook.ipynb --hw 1 --json
```

## Student function names (HW4)

Your notebook must define:

- `build_adjacency_matrix(edges, n_nodes)`
- `gcn_forward(x, adj, weights, biases)`
- `neighborhood_sample(adj, target, n_sample, seed=0)`
- `gat_forward(x, adj, weight, attention)`

See `samples/CSC8851_S2026_HW4_Sample.ipynb` for a working reference.
