from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import build_vocab
from stream_dataset import StreamingPrimingDataset, count_split_labels
from train_model import BiLSTMClassifier, BinaryFocalLossWithLogits, CNNClassifier, compute_binary_metrics

PBAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} samples [{elapsed}<{remaining}, {rate_fmt}{postfix}]"


def resolve_progress(mode: str) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    return sys.stderr.isatty()


def evaluate_stream(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    max_batches: Optional[int] = None,
    total_samples: Optional[int] = None,
    desc: str = "val",
    show_progress: bool = True,
) -> Tuple[float, Dict[str, float], int]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    logits_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []
    with torch.no_grad():
        pbar = tqdm(
            total=total_samples,
            desc=desc,
            unit="sample",
            leave=False,
            dynamic_ncols=True,
            bar_format=PBAR_FORMAT,
            disable=not show_progress,
        )
        for bi, (x, y) in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            bs = x.size(0)
            total_loss += loss.item() * bs
            total_count += bs
            logits_all.append(logits.detach().cpu())
            y_all.append(y.detach().cpu())
            pbar.update(bs)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.close()

    if total_count == 0:
        empty = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "specificity": 0.0, "balanced_acc": 0.0, "f1": 0.0}
        return 0.0, empty, 0
    logits_cat = torch.cat(logits_all, dim=0)
    y_cat = torch.cat(y_all, dim=0)
    metrics = compute_binary_metrics(logits_cat, y_cat)
    return total_loss / total_count, metrics, total_count


def make_loader(dataset: StreamingPrimingDataset, batch_size: int, num_workers: int, prefetch_factor: int) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream training for very large BED files (e.g. 30GB).")
    parser.add_argument("--bed", type=Path, default=Path("../demo/demo.bed"), help="Path to BED file.")
    parser.add_argument("--seq-len", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["rnn", "cnn"], default="rnn", help="Model backbone.")
    parser.add_argument("--num-workers", type=int, default=4, help="Use >0 for parallel data preprocessing.")
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--use-one-hot", action="store_true")
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Debug only: cap train batches/epoch.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Cap validation batches/epoch for speed.")
    parser.add_argument("--save-model", type=Path, default=Path("./internal_priming_stream.pt"))
    parser.add_argument("--progress", choices=["auto", "on", "off"], default="auto", help="Progress bar display mode.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    show_progress = resolve_progress(args.progress)
    vocab = build_vocab()

    counts = count_split_labels(args.bed, train_ratio=args.train_ratio, seed=args.seed)
    train_pos = counts["train"]["pos"]
    train_neg = counts["train"]["neg"]
    val_pos = counts["val"]["pos"]
    val_neg = counts["val"]["neg"]
    total = train_pos + train_neg + val_pos + val_neg
    if train_pos == 0 or train_neg == 0:
        raise ValueError("Train split has only one class. Adjust --train-ratio or verify labels.")
    if val_pos == 0 or val_neg == 0:
        raise ValueError("Val split has only one class. Adjust --train-ratio or verify labels.")

    print(
        f"Total samples={total} | "
        f"Train(pos={train_pos}, neg={train_neg}) | "
        f"Val(pos={val_pos}, neg={val_neg})"
    )

    train_ds = StreamingPrimingDataset(
        bed_path=args.bed,
        seq_len=args.seq_len,
        split="train",
        train_ratio=args.train_ratio,
        seed=args.seed,
        vocab=vocab,
    )
    val_ds = StreamingPrimingDataset(
        bed_path=args.bed,
        seq_len=args.seq_len,
        split="val",
        train_ratio=args.train_ratio,
        seed=args.seed,
        vocab=vocab,
    )
    train_loader = make_loader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    val_loader = make_loader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)

    if args.model == "cnn":
        model = CNNClassifier(
            vocab_size=len(vocab),
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            use_one_hot=args.use_one_hot,
        ).to(device)
    else:
        model = BiLSTMClassifier(
            vocab_size=len(vocab),
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_one_hot=args.use_one_hot,
        ).to(device)

    pos_weight = torch.tensor([train_neg / train_pos], dtype=torch.float32, device=device)
    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = BinaryFocalLossWithLogits(gamma=args.focal_gamma, pos_weight=pos_weight)
    print(f"Using model={args.model} loss={args.loss} pos_weight={pos_weight.item():.4f} num_workers={args.num_workers}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_bal_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        train_logits_all: List[torch.Tensor] = []
        train_y_all: List[torch.Tensor] = []

        train_total_samples = train_pos + train_neg
        if args.max_train_batches is not None:
            train_total_samples = min(train_total_samples, args.max_train_batches * args.batch_size)
        train_pbar = tqdm(
            total=train_total_samples,
            desc=f"epoch {epoch}/{args.epochs} train",
            unit="sample",
            leave=False,
            dynamic_ncols=True,
            bar_format=PBAR_FORMAT,
            disable=not show_progress,
        )
        for bi, (x, y) in enumerate(train_loader):
            if args.max_train_batches is not None and bi >= args.max_train_batches:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            train_loss += loss.item() * bs
            train_count += bs
            train_logits_all.append(logits.detach().cpu())
            train_y_all.append(y.detach().cpu())
            train_pbar.update(bs)
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_pbar.close()

        if train_count == 0:
            raise RuntimeError("No train samples consumed in this epoch.")
        train_loss /= train_count
        train_metrics = compute_binary_metrics(torch.cat(train_logits_all, dim=0), torch.cat(train_y_all, dim=0))

        val_loss, val_metrics, val_count = evaluate_stream(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            max_batches=args.max_val_batches,
            total_samples=min(val_pos + val_neg, args.max_val_batches * args.batch_size)
            if args.max_val_batches is not None
            else (val_pos + val_neg),
            desc=f"epoch {epoch}/{args.epochs} val",
            show_progress=show_progress,
        )
        if val_count == 0:
            raise RuntimeError("No val samples consumed in this epoch.")

        if val_metrics["balanced_acc"] > best_val_bal_acc:
            best_val_bal_acc = val_metrics["balanced_acc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "seq_len": args.seq_len,
                    "args": vars(args),
                },
                args.save_model,
            )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_bal_acc={train_metrics['balanced_acc']:.4f} "
            f"train_prec={train_metrics['precision']:.4f} train_rec={train_metrics['recall']:.4f} train_f1={train_metrics['f1']:.4f} | "
            f"val_loss={val_loss:.4f} val_bal_acc={val_metrics['balanced_acc']:.4f} "
            f"val_prec={val_metrics['precision']:.4f} val_rec={val_metrics['recall']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_spec={val_metrics['specificity']:.4f}"
        )

    print(f"Best val_balanced_acc={best_val_bal_acc:.4f}")
    print(f"Saved best model to: {args.save_model.resolve()}")


if __name__ == "__main__":
    main()
