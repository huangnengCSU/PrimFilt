from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from dataset import build_vocab, encode_seq, parse_bed_line
from train_model import BiLSTMClassifier, CNNClassifier

PBAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} samples [{elapsed}<{remaining}, {rate_fmt}]"


def resolve_progress(mode: str) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    return sys.stderr.isatty()


class BedInferenceDataset(Dataset):
    def __init__(self, rows: Sequence[List[str]], seq_len: int, vocab: Dict[str, int]) -> None:
        self.rows = rows
        self.seq_len = seq_len
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.rows[idx]
        read_seq = row[5]
        ref_seq = row[6]
        read_ids = encode_seq(read_seq, self.seq_len, self.vocab)
        ref_ids = encode_seq(ref_seq, self.seq_len, self.vocab)
        x = torch.tensor([read_ids, ref_ids], dtype=torch.long)
        return x, idx


def load_valid_rows(path: Path) -> Tuple[List[List[str]], List[str]]:
    valid_rows: List[List[str]] = []
    passthrough: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw or raw.startswith("#"):
                passthrough.append(raw)
                continue
            parts = raw.split("\t")
            if len(parts) < 7:
                passthrough.append(raw)
                continue
            try:
                _ = parse_bed_line(raw)
            except Exception:
                passthrough.append(raw)
                continue
            valid_rows.append(parts)
    return valid_rows, passthrough


def build_model_from_checkpoint(ckpt: dict, device: torch.device) -> nn.Module:
    args = ckpt.get("args", {})
    vocab = ckpt.get("vocab", build_vocab())
    model_type = args.get("model", "rnn")
    use_one_hot = args.get("use_one_hot", False)
    embed_dim = args.get("embed_dim", 16)
    hidden_dim = args.get("hidden_dim", 64)
    num_layers = args.get("num_layers", 2)
    dropout = args.get("dropout", 0.2)

    if model_type == "cnn":
        model = CNNClassifier(
            vocab_size=len(vocab),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_one_hot=use_one_hot,
        )
    else:
        model = BiLSTMClassifier(
            vocab_size=len(vocab),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_one_hot=use_one_hot,
        )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_probs(
    model: nn.Module,
    rows: Sequence[List[str]],
    seq_len: int,
    vocab: Dict[str, int],
    batch_size: int,
    device: torch.device,
    show_progress: bool,
) -> List[float]:
    dataset = BedInferenceDataset(rows=rows, seq_len=seq_len, vocab=vocab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs = [0.0] * len(rows)
    pbar = tqdm(
        total=len(rows),
        desc="predict",
        unit="sample",
        leave=False,
        dynamic_ncols=True,
        bar_format=PBAR_FORMAT,
        disable=not show_progress,
    )
    with torch.no_grad():
        for x, batch_idx in loader:
            x = x.to(device)
            logits = model(x)
            p = torch.sigmoid(logits).squeeze(1).detach().cpu().tolist()
            for i, row_idx in enumerate(batch_idx.tolist()):
                probs[row_idx] = float(p[i])
            pbar.update(len(batch_idx))
    pbar.close()
    return probs


def format_out_row(parts: List[str], prob_pos: float, threshold: float, prob_cols: int) -> str:
    pred_label = "1" if prob_pos >= threshold else "0"
    out = list(parts)
    out[4] = pred_label
    insert_at = 7  # after ref_seq (index 6)
    if prob_cols == 1:
        out.insert(insert_at, f"{prob_pos:.6f}")
    else:
        out.insert(insert_at, f"{1.0 - prob_pos:.6f}")
        out.insert(insert_at + 1, f"{prob_pos:.6f}")
    return "\t".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict BED labels and write probabilities.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to trained checkpoint (.pt).")
    parser.add_argument("--input-bed", type=Path, required=True, help="Input BED path.")
    parser.add_argument("--output-bed", type=Path, required=True, help="Output BED path.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for label=1.")
    parser.add_argument(
        "--prob-cols",
        type=int,
        choices=[1, 2],
        default=2,
        help="Append 1 column (P(label=1)) or 2 columns (P(label=0), P(label=1)).",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--progress", choices=["auto", "on", "off"], default="auto", help="Progress bar display mode.")
    args = parser.parse_args()
    show_progress = resolve_progress(args.progress)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    seq_len = int(ckpt.get("seq_len", 240))
    vocab = ckpt.get("vocab", build_vocab())
    model = build_model_from_checkpoint(ckpt=ckpt, device=device)

    valid_rows, passthrough = load_valid_rows(args.input_bed)
    probs = predict_probs(
        model=model,
        rows=valid_rows,
        seq_len=seq_len,
        vocab=vocab,
        batch_size=args.batch_size,
        device=device,
        show_progress=show_progress,
    )

    # Preserve file line order by second pass write.
    prob_iter = iter(probs)
    total_lines = len(valid_rows) + len(passthrough)
    write_pbar = tqdm(
        total=total_lines,
        desc="write",
        unit="line",
        leave=False,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}行 [{elapsed}<{remaining}, {rate_fmt}]",
        disable=not show_progress,
    )
    with args.input_bed.open("r", encoding="utf-8", errors="ignore") as src, args.output_bed.open("w", encoding="utf-8") as dst:
        for line in src:
            raw = line.rstrip("\n")
            if not raw or raw.startswith("#"):
                dst.write(raw + "\n")
                write_pbar.update(1)
                continue
            parts = raw.split("\t")
            if len(parts) < 7:
                dst.write(raw + "\n")
                write_pbar.update(1)
                continue
            try:
                _ = parse_bed_line(raw)
            except Exception:
                dst.write(raw + "\n")
                write_pbar.update(1)
                continue
            p = next(prob_iter)
            out_line = format_out_row(parts, prob_pos=p, threshold=args.threshold, prob_cols=args.prob_cols)
            dst.write(out_line + "\n")
            write_pbar.update(1)
    write_pbar.close()

    print(f"Loaded model: {args.model_path.resolve()}")
    print(f"Input rows predicted: {len(valid_rows)}")
    print(f"Wrote output BED: {args.output_bed.resolve()}")
    if args.prob_cols == 1:
        print("Added column after ref_seq: P(label=1)")
    else:
        print("Added columns after ref_seq: P(label=0), P(label=1)")


if __name__ == "__main__":
    main()
