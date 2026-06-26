"""Temporal train/shadow split for Polymarket corpus labels."""

from __future__ import annotations

from bigan.v8.polymarket.corpus.contracts import (
    PolymarketCorpusBuildConfig,
    PolymarketCorpusLabelRow,
    PolymarketCorpusSplit,
    stable_hash,
)


def build_polymarket_train_shadow_split(
    *,
    label_rows: tuple[PolymarketCorpusLabelRow, ...],
    config: PolymarketCorpusBuildConfig,
) -> PolymarketCorpusSplit:
    """Build a deterministic temporal split with no cross-time leakage."""

    if not label_rows:
        raise ValueError("label_rows must not be empty")
    decision_times = sorted({row.decision_ts for row in label_rows})
    if len(decision_times) < 2:
        raise ValueError("at least two decision timestamps are required")
    split_index = int(len(decision_times) * config.train_fraction)
    split_index = min(max(1, split_index), len(decision_times) - 1)
    split_ts = decision_times[split_index]
    train = tuple(row for row in label_rows if row.decision_ts < split_ts)
    shadow = tuple(row for row in label_rows if row.decision_ts >= split_ts)
    train_hash = stable_hash([row.to_dict() for row in train])
    shadow_hash = stable_hash([row.to_dict() for row in shadow])
    return PolymarketCorpusSplit(
        split_ts=split_ts,
        split_hash=stable_hash(
            {
                "split_ts": split_ts,
                "train_dataset_hash": train_hash,
                "shadow_dataset_hash": shadow_hash,
                "train_fraction": config.train_fraction,
            }
        ),
        train_label_count=len(train),
        shadow_label_count=len(shadow),
        max_train_decision_ts=max(row.decision_ts for row in train),
        min_shadow_decision_ts=min(row.decision_ts for row in shadow),
        train_dataset_hash=train_hash,
        shadow_dataset_hash=shadow_hash,
    )
