"""Canonical dataset name for target-reference source-to-target training."""

from .dosc_unaligned_dataset import DoscUnalignedDataset


class TrscUnalignedDataset(DoscUnalignedDataset):
    """Canonical alias that preserves the strict source-label protocol."""
