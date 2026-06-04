"""Config-driven training entrypoint for LightGCN experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

from lightgcnrec.data import edges_from_interactions, load_csv_records, split_edges
from lightgcnrec.experiments.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML or JSON experiment config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and data paths without training.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dry_run:
        validate_config(config)
        print(f"Validated config: {config.get('experiment', {}).get('name', args.config)}")
        return
    run_training(config)


def validate_config(config: dict) -> None:
    required_sections = ["dataset", "model", "training", "output"]
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")
    dataset = config["dataset"]
    if dataset.get("synthetic"):
        return
    interactions_path = Path(dataset["interactions_path"])
    if not interactions_path.exists():
        raise FileNotFoundError(f"Dataset not found: {interactions_path}")


def run_training(config: dict) -> None:
    """Run a small LightGCN training job from config."""

    validate_config(config)
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install torch to run training.") from exc

    from lightgcnrec.models.lightgcn import LightGCN
    from lightgcnrec.models.losses import bpr_loss

    dataset = config["dataset"]
    if dataset.get("synthetic"):
        edges = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 2)]
        num_users = 3
        num_items = 3
    else:
        records = load_csv_records(dataset["interactions_path"])
        edges, user_mapping, item_mapping = edges_from_interactions(
            records,
            dataset["user_col"],
            dataset["item_col"],
            dataset.get("rating_col"),
            dataset.get("rating_threshold"),
        )
        num_users = len(user_mapping)
        num_items = len(item_mapping)

    train_edges, _, _ = split_edges(
        edges,
        val_ratio=float(config["training"].get("val_ratio", 0.1)),
        test_ratio=float(config["training"].get("test_ratio", 0.1)),
        seed=int(config["training"].get("seed", 42)),
    )
    device = torch.device(config["training"].get("device", "cpu"))
    edge_index = torch.tensor(train_edges, dtype=torch.long, device=device).t().contiguous()
    model = LightGCN(
        num_users,
        num_items,
        int(config["model"].get("embedding_dim", 64)),
        int(config["model"].get("num_layers", 3)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"].get("lr", 0.001)))
    epochs = int(config["training"].get("epochs", 1))
    batch_size = int(config["training"].get("batch_size", 1024))
    rng = random.Random(int(config["training"].get("seed", 42)))
    all_items = set(range(num_items))

    for epoch in range(epochs):
        batch = [rng.choice(train_edges) for _ in range(min(batch_size, len(train_edges)))]
        users = torch.tensor([u for u, _ in batch], dtype=torch.long, device=device)
        pos_items = torch.tensor([i for _, i in batch], dtype=torch.long, device=device)
        neg_items = torch.tensor([rng.choice(tuple(all_items - {i})) for _, i in batch], dtype=torch.long, device=device)
        user_emb, item_emb = model(edge_index)
        loss = bpr_loss(
            user_emb[users],
            model.users_emb(users),
            item_emb[pos_items],
            model.items_emb(pos_items),
            item_emb[neg_items],
            model.items_emb(neg_items),
            float(config["training"].get("lambda_reg", 1e-4)),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch={epoch + 1} loss={loss.item():.6f}")

    output_dir = Path(config["output"].get("dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    print(f"Saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
