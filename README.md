# Toward Diversified Graph Recommendation via Semantic and Topology Augmentation With LLMs

This repository contains the research code for the IEEE Access paper:

> Zachary S. Pangan, Shaowen Peng, Shoko Wakamiya, and Eiji Aramaki. “Toward Diversified Graph Recommendation via Semantic and Topology Augmentation With LLMs.” IEEE Access, 2025. DOI: `10.1109/ACCESS.2025.3637140`.

The project studies model-agnostic augmentation for graph recommendation. It uses LightGCN as the base recommender and adds two kinds of LLM-guided augmentation:

- Semantic augmentation: prompt-derived user/item text representations and fusion with graph embeddings.
- Topological augmentation: prompt-derived user-item and item-item links constrained by proximal categories.

The primary experiments are on MovieLens-1M and UCSD Steam. The repository is organized as a hybrid package: reusable code lives in `lightgcnrec/`, while cleaned notebooks in `notebooks/` document the original study workflow.

## Repository Layout

```text
lightgcnrec/          Reusable data, model, metric, augmentation, and experiment modules
configs/              YAML experiment configs for paper-style runs
prompts/              Versioned prompt templates used by augmentation stages
notebooks/            Cleaned example notebooks
notebooks/archive/    Legacy exploratory notebooks retained for provenance
data/README.md        Dataset download and local layout instructions
tests/                Lightweight unit and smoke tests
```

## Installation

Create an environment with Python 3.10 or newer, then install the project.

```bash
pip install -e ".[dev]"
```

For full training, install the training and visualization extras in an environment compatible with your CUDA/PyTorch setup.

```bash
pip install -e ".[train,llm,viz,dev]"
```

PyTorch Geometric wheels are platform-specific; if installation fails, follow the official PyG instructions for your Torch/CUDA version.

## Data

Raw datasets, generated embeddings, model checkpoints, and full outputs are intentionally not tracked. See `data/README.md` for the expected local layout.

Supported public datasets:

- MovieLens-1M
- UCSD Steam Video Game and Bundle Data

## Running Experiments

Validate a config without requiring the dataset:

```bash
python -m lightgcnrec.experiments.train --config configs/smoke_synthetic.yaml --dry-run
```

Run the tiny synthetic smoke experiment:

```bash
python -m lightgcnrec.experiments.train --config configs/smoke_synthetic.yaml
```

Paper-style entrypoints:

```bash
python -m lightgcnrec.experiments.train --config configs/ml1m_baseline.yaml
python -m lightgcnrec.experiments.train --config configs/steam_topological.yaml
python -m lightgcnrec.experiments.evaluate --recommendations outputs/recommendations.json --ground-truth outputs/ground_truth.json --k 20
```

The configs define dataset paths, model settings, augmentation mode, seed, device, output directory, and metric K values. Adjust local paths under `dataset:` before full runs.

## Notebooks

- `notebooks/ml1m_experiment.ipynb`: MovieLens-1M experiment workflow.
- `notebooks/steam_experiment.ipynb`: Steam experiment workflow.
- `notebooks/visualize_embeddings.ipynb`: embedding and metric visualization workflow.

Notebook outputs are stripped for clean diffs and archival safety. Re-run notebooks only after preparing the external data and optional augmentation caches.

## Citation

Use the metadata in `CITATION.cff` or cite the paper DOI:

```bibtex
@article{pangan2025diversified,
  title={Toward Diversified Graph Recommendation via Semantic and Topology Augmentation With LLMs},
  author={Pangan, Zachary S. and Peng, Shaowen and Wakamiya, Shoko and Aramaki, Eiji},
  journal={IEEE Access},
  year={2025},
  doi={10.1109/ACCESS.2025.3637140}
}
```

## License

This repository is released under the MIT License.
