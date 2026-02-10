from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<PAD>"
DEFAULT_ALPHABET = [PAD_TOKEN, "A", "C", "G", "T", "N", "-", "*"]


def build_vocab(alphabet: Sequence[str] = DEFAULT_ALPHABET) -> Dict[str, int]:
    return {token: idx for idx, token in enumerate(alphabet)}


def normalize_seq(seq: str) -> str:
    return seq.strip().upper()


def encode_seq(seq: str, seq_len: int, vocab: Dict[str, int], unk_token: str = "N") -> List[int]:
    unk_id = vocab[unk_token]
    pad_id = vocab[PAD_TOKEN]
    seq = normalize_seq(seq)
    encoded: List[int] = []
    for ch in seq[:seq_len]:
        # In this project, '*' means no base (padding-like), so map it to PAD.
        if ch == "*":
            encoded.append(pad_id)
        else:
            encoded.append(vocab.get(ch, unk_id))
    if len(encoded) < seq_len:
        encoded.extend([pad_id] * (seq_len - len(encoded)))
    return encoded


@dataclass
class BedRecord:
    chrom: str
    start: int
    end: int
    read_name: str
    label: int
    read_seq: str
    ref_seq: str


def parse_bed_line(line: str) -> BedRecord:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 7:
        raise ValueError(f"BED line needs >= 7 columns, got {len(parts)}: {line[:120]}")
    return BedRecord(
        chrom=parts[0],
        start=int(parts[1]),
        end=int(parts[2]),
        read_name=parts[3],
        label=int(parts[4]),
        read_seq=parts[5],
        ref_seq=parts[6],
    )


def load_bed_records(path: Path) -> List[BedRecord]:
    records: List[BedRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip() or raw.startswith("#"):
                continue
            records.append(parse_bed_line(raw))
    return records


class PrimingDataset(Dataset):
    """Return x as LongTensor[2, seq_len], y as FloatTensor[1]."""

    def __init__(self, records: Sequence[BedRecord], seq_len: int = 240, vocab: Dict[str, int] | None = None):
        self.records = list(records)
        self.seq_len = seq_len
        self.vocab = vocab or build_vocab()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        read_ids = encode_seq(record.read_seq, self.seq_len, self.vocab)
        ref_ids = encode_seq(record.ref_seq, self.seq_len, self.vocab)
        x = torch.tensor([read_ids, ref_ids], dtype=torch.long)
        y = torch.tensor([float(record.label)], dtype=torch.float32)
        return x, y
