from pathlib import Path
import argparse
import json
import random

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

from MANNDataset import build_mann_dataloaders
from MANNModel import MANN, mann_mse_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    return {
        "x_main": batch["x_main"].to(device, non_blocking=True),
        "x_gate": batch["x_gate"].to(device, non_blocking=True),
        "y": batch["y"].to(device, non_blocking=True),
        "action_id": batch["action_id"].to(device, non_blocking=True),
        "frame_index": batch["frame_index"].to(device, non_blocking=True),
        "clip_name": batch["clip_name"],
    }


def run_epoch(model, dataloader, optimizer, device, training):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    progress = tqdm(dataloader, desc="train" if training else "val", leave=False)

    for batch in progress:
        batch = move_batch_to_device(batch, device)

        with torch.set_grad_enabled(training):
            y_pred = model(batch["x_main"], batch["x_gate"])
            loss = mann_mse_loss(y_pred, batch["y"])

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = batch["x_main"].shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{float(loss.detach()):.6f}")

    if total_samples == 0:
        return 0.0
    return total_loss / total_samples


def save_checkpoint(path, model, optimizer, epoch, best_val_loss, spec, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "best_val_loss": float(best_val_loss),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model.config.__dict__,
            "data_spec": {
                "stage": spec.stage,
                "x_main_dim": spec.x_main_dim,
                "x_gate_dim": spec.x_gate_dim,
                "y_dim": spec.y_dim,
                "action_labels": list(spec.action_labels),
            },
            "train_args": vars(args),
        },
        path,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train a PyTorch MANN model from exported MANN databases.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the exported MANN database (.npz).")
    parser.add_argument("--output-dir", type=Path, default=Path("output/mann/checkpoints"), help="Directory for checkpoints and logs.")
    parser.add_argument("--split-path", type=Path, default=None, help="Optional JSON path for clip-level train/val/test splits.")
    parser.add_argument("--stats-path", type=Path, default=None, help="Optional NPZ path for normalization statistics.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    parser.add_argument("--num-experts", type=int, default=8, help="Number of MANN experts.")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension of the expert MLP.")
    parser.add_argument("--gating-hidden-dim", type=int, default=32, help="Hidden dimension of the gating network.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability used in gating and expert MLP layers.")
    parser.add_argument("--loader-workers", type=int, default=0, help="PyTorch DataLoader worker count.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio when split file is not provided.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio when split file is not provided.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override, e.g. cpu, cuda, mps.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable dataset normalization.")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable DataLoader pin_memory.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(
        args.device if args.device is not None else (
            "cuda" if torch.cuda.is_available() else (
                "mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else "cpu"
            )
        )
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.stats_path or output_dir / "stats.npz"
    split_path = args.split_path or output_dir / "splits.json"

    dataloaders = build_mann_dataloaders(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        split_path=split_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        normalize=not args.no_normalize,
        stats_path=stats_path,
        num_workers=args.loader_workers,
        pin_memory=not args.no_pin_memory,
    )
    spec = dataloaders["spec"]

    model = MANN.from_data_spec(
        spec,
        hidden_dim=args.hidden_dim,
        gating_hidden_dim=args.gating_hidden_dim,
        num_experts=args.num_experts,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = run_epoch(model, dataloaders["train"], optimizer, device, training=True)
        val_loader = dataloaders["val"]
        val_loss = run_epoch(model, val_loader, optimizer, device, training=False) if len(val_loader.dataset) > 0 else train_loss

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )
        print(f"  train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, best_val_loss, spec, args)
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_val_loss, spec, args)

    with (output_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    print(f"Training finished. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints written to {output_dir}")


if __name__ == "__main__":
    main()
