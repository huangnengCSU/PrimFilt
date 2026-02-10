from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterator, Tuple

import torch
from torch.utils.data import IterableDataset, get_worker_info

from dataset import PAD_TOKEN, build_vocab, encode_seq, parse_bed_line


def stable_u64(s: str, seed: int = 42) -> int:
    h = hashlib.blake2b(digest_size=8, person=seed.to_bytes(8, byteorder="little", signed=False))
    h.update(s.encode("utf-8", errors="ignore"))
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


def in_train_split(read_name: str, train_ratio: float, seed: int) -> bool:
    # deterministic split in [0, 1)
    u = stable_u64(read_name, seed=seed)
    frac = (u % 10_000_000) / 10_000_000.0
    return frac < train_ratio


class StreamingPrimingDataset(IterableDataset):
    """
    Stream BED lines and emit:
      x: LongTensor [2, seq_len]
      y: FloatTensor [1]

    Worker sharding is byte-range based so each worker reads a different file chunk.
    """

    def __init__(
        self,
        bed_path: Path,
        seq_len: int = 240,
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
        vocab: Dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        self.bed_path = bed_path
        self.seq_len = seq_len
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.vocab = vocab or build_vocab()
        self.pad_id = self.vocab[PAD_TOKEN]

    def _want_record(self, read_name: str) -> bool:
        train_side = in_train_split(read_name=read_name, train_ratio=self.train_ratio, seed=self.seed)
        return train_side if self.split == "train" else (not train_side)

    def _iter_worker_chunk(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        file_size = self.bed_path.stat().st_size
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        chunk_size = (file_size + num_workers - 1) // num_workers
        start = worker_id * chunk_size
        end = min(file_size, (worker_id + 1) * chunk_size)

        with self.bed_path.open("rb") as f:
            if start > 0:
                f.seek(start - 1)
                _ = f.readline()  # drop partial line
            else:
                f.seek(0)

            while f.tell() < end:
                raw = f.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    rec = parse_bed_line(line)
                except Exception:
                    continue
                if not self._want_record(rec.read_name):
                    continue
                read_ids = encode_seq(rec.read_seq, self.seq_len, self.vocab)
                ref_ids = encode_seq(rec.ref_seq, self.seq_len, self.vocab)
                x = torch.tensor([read_ids, ref_ids], dtype=torch.long)
                y = torch.tensor([float(rec.label)], dtype=torch.float32)
                yield x, y

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        return self._iter_worker_chunk()


def count_split_labels(bed_path: Path, train_ratio: float, seed: int) -> Dict[str, Dict[str, int]]:
    counts = {"train": {"pos": 0, "neg": 0}, "val": {"pos": 0, "neg": 0}}
    with bed_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = parse_bed_line(line)
            except Exception:
                continue
            split = "train" if in_train_split(rec.read_name, train_ratio=train_ratio, seed=seed) else "val"
            if rec.label == 1:
                counts[split]["pos"] += 1
            else:
                counts[split]["neg"] += 1
    return counts
