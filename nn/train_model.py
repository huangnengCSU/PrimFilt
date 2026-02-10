from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm.auto import tqdm

from dataset import PrimingDataset, build_vocab, load_bed_records

PBAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} samples [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
PBAR_NCOLS = 100


def resolve_progress(mode: str) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    return sys.stderr.isatty()


class BiLSTMClassifier(nn.Module):
    """
    Input: LongTensor [batch, 2, seq_len], containing token ids for read/ref.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        padding_idx: int = 0,
        use_one_hot: bool = False,
    ) -> None:
        super().__init__()
        self.use_one_hot = use_one_hot
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        if not use_one_hot:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
            lstm_input_size = embed_dim * 2
        else:
            self.embedding = None
            lstm_input_size = vocab_size * 2

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(self, ids: torch.Tensor) -> torch.Tensor:
        if self.use_one_hot:
            return torch.nn.functional.one_hot(ids, num_classes=self.vocab_size).float()
        assert self.embedding is not None
        return self.embedding(ids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        read_ids = x[:, 0, :]
        ref_ids = x[:, 1, :]
        read_feat = self._encode(read_ids)
        ref_feat = self._encode(ref_ids)
        feat = torch.cat([read_feat, ref_feat], dim=-1)  # [B, L, 2*embed] or [B, L, 2*vocab]

        _, (h_n, _) = self.lstm(feat)
        # last layer forward/backward hidden states
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h = torch.cat([h_fwd, h_bwd], dim=-1)
        return self.classifier(h)


class CNNClassifier(nn.Module):
    """
    Input: LongTensor [batch, 2, seq_len], containing token ids for read/ref.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        padding_idx: int = 0,
        use_one_hot: bool = False,
    ) -> None:
        super().__init__()
        self.use_one_hot = use_one_hot
        self.vocab_size = vocab_size
        if not use_one_hot:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
            in_channels = embed_dim * 2
        else:
            self.embedding = None
            in_channels = vocab_size * 2

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=128, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(self, ids: torch.Tensor) -> torch.Tensor:
        if self.use_one_hot:
            return torch.nn.functional.one_hot(ids, num_classes=self.vocab_size).float()
        assert self.embedding is not None
        return self.embedding(ids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        read_ids = x[:, 0, :]
        ref_ids = x[:, 1, :]
        read_feat = self._encode(read_ids)
        ref_feat = self._encode(ref_ids)
        feat = torch.cat([read_feat, ref_feat], dim=-1)  # [B, L, C]
        feat = feat.transpose(1, 2)  # [B, C, L] for Conv1d
        feat = self.conv(feat)
        return self.classifier(feat)


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight)
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - pt).pow(self.gamma)
        return (focal_weight * bce).mean()


def stratified_split_indices(labels: Sequence[int], train_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    pos_idx = [i for i, y in enumerate(labels) if y == 1]
    neg_idx = [i for i, y in enumerate(labels) if y == 0]
    if not pos_idx or not neg_idx:
        raise ValueError("Need both positive and negative samples for stratified split.")

    g = torch.Generator().manual_seed(seed)
    pos_perm = torch.randperm(len(pos_idx), generator=g).tolist()
    neg_perm = torch.randperm(len(neg_idx), generator=g).tolist()
    pos_idx = [pos_idx[i] for i in pos_perm]
    neg_idx = [neg_idx[i] for i in neg_perm]

    pos_train = max(1, int(len(pos_idx) * train_ratio))
    neg_train = max(1, int(len(neg_idx) * train_ratio))
    if pos_train >= len(pos_idx):
        pos_train = len(pos_idx) - 1
    if neg_train >= len(neg_idx):
        neg_train = len(neg_idx) - 1

    train_indices = pos_idx[:pos_train] + neg_idx[:neg_train]
    val_indices = pos_idx[pos_train:] + neg_idx[neg_train:]
    train_perm = torch.randperm(len(train_indices), generator=g).tolist()
    val_perm = torch.randperm(len(val_indices), generator=g).tolist()
    train_indices = [train_indices[i] for i in train_perm]
    val_indices = [val_indices[i] for i in val_perm]
    return train_indices, val_indices


def compute_binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    y = labels.float()

    tp = ((preds == 1) & (y == 1)).sum().item()
    tn = ((preds == 0) & (y == 0)).sum().item()
    fp = ((preds == 1) & (y == 0)).sum().item()
    fn = ((preds == 0) & (y == 1)).sum().item()

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_acc = 0.5 * (recall + specificity)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_acc": balanced_acc,
        "f1": f1,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    total_samples: Optional[int] = None,
    desc: str = "val",
    show_progress: bool = True,
) -> Tuple[float, Dict[str, float]]:
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
            ncols=PBAR_NCOLS,
            bar_format=PBAR_FORMAT,
            disable=not show_progress,
        )
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_count += x.size(0)
            logits_all.append(logits.detach().cpu())
            y_all.append(y.detach().cpu())
            pbar.update(x.size(0))
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.close()
    if not logits_all:
        return 0.0, {"acc": 0.0, "precision": 0.0, "recall": 0.0, "specificity": 0.0, "balanced_acc": 0.0, "f1": 0.0}
    logits_cat = torch.cat(logits_all, dim=0)
    y_cat = torch.cat(y_all, dim=0)
    metrics = compute_binary_metrics(logits_cat, y_cat)
    return total_loss / max(total_count, 1), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RNN/CNN baseline for internal priming classification.")
    parser.add_argument("--bed", type=Path, default=Path("../demo/demo.bed"), help="Path to BED file.")
    parser.add_argument("--seq-len", type=int, default=240, help="Sequence length after truncate/pad.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["rnn", "cnn"], default="rnn", help="Model backbone.")
    parser.add_argument("--use-one-hot", action="store_true", help="Use one-hot instead of embedding.")
    parser.add_argument(
        "--sampler",
        choices=["weighted", "none"],
        default="weighted",
        help="Weighted sampler is recommended for highly imbalanced data.",
    )
    parser.add_argument(
        "--loss",
        choices=["bce", "focal"],
        default="bce",
        help="Use focal loss to further focus on hard minority samples.",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma for focal loss.")
    parser.add_argument("--save-model", type=Path, default=Path("./internal_priming_model.pt"))
    parser.add_argument("--progress", choices=["auto", "on", "off"], default="auto", help="Progress bar display mode.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    show_progress = resolve_progress(args.progress)

    records = load_bed_records(args.bed)
    if len(records) < 2:
        raise ValueError(f"Not enough records in {args.bed}.")
    labels = [r.label for r in records]
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    if total_pos == 0 or total_neg == 0:
        raise ValueError("Dataset must contain both positive and negative labels.")

    vocab = build_vocab()
    dataset = PrimingDataset(records=records, seq_len=args.seq_len, vocab=vocab)
    train_indices, val_indices = stratified_split_indices(labels=labels, train_ratio=args.train_ratio, seed=args.seed)
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_pos = sum(labels[i] for i in train_indices)
    train_neg = len(train_indices) - train_pos
    val_pos = sum(labels[i] for i in val_indices)
    val_neg = len(val_indices) - val_pos

    sampler = None
    if args.sampler == "weighted":
        class_counts = torch.tensor([train_neg, train_pos], dtype=torch.float32)
        class_weights = torch.where(class_counts > 0, 1.0 / class_counts, torch.zeros_like(class_counts))
        sample_weights = [class_weights[labels[i]].item() for i in train_indices]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

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

    pos_weight = None
    if train_pos > 0:
        pos_weight = torch.tensor([train_neg / train_pos], dtype=torch.float32, device=device)

    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = BinaryFocalLossWithLogits(gamma=args.focal_gamma, pos_weight=pos_weight)

    print(
        f"Total samples={len(labels)} (pos={total_pos}, neg={total_neg}) | "
        f"Train={len(train_indices)} (pos={train_pos}, neg={train_neg}) | "
        f"Val={len(val_indices)} (pos={val_pos}, neg={val_neg})"
    )
    if pos_weight is not None:
        print(f"Using model={args.model}, pos_weight={pos_weight.item():.4f}, sampler={args.sampler}, loss={args.loss}")
    else:
        print(f"Using model={args.model}, sampler={args.sampler}, loss={args.loss}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_balanced_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        train_logits_all: List[torch.Tensor] = []
        train_y_all: List[torch.Tensor] = []
        train_pbar = tqdm(
            total=len(train_indices),
            desc=f"epoch {epoch}/{args.epochs} train",
            unit="sample",
            leave=False,
            ncols=PBAR_NCOLS,
            bar_format=PBAR_FORMAT,
            disable=not show_progress,
        )
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_count += x.size(0)
            train_logits_all.append(logits.detach().cpu())
            train_y_all.append(y.detach().cpu())
            train_pbar.update(x.size(0))
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_pbar.close()

        train_loss /= max(train_count, 1)
        train_metrics = compute_binary_metrics(torch.cat(train_logits_all, dim=0), torch.cat(train_y_all, dim=0))
        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
            total_samples=len(val_indices),
            desc=f"epoch {epoch}/{args.epochs} val",
            show_progress=show_progress,
        )

        if val_metrics["balanced_acc"] > best_val_balanced_acc:
            best_val_balanced_acc = val_metrics["balanced_acc"]
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

    print(f"Best val_balanced_acc={best_val_balanced_acc:.4f}")
    print(f"Saved best model to: {args.save_model.resolve()}")


if __name__ == "__main__":
    main()
