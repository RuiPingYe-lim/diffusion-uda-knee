"""Target-reference dataset with optional multi-reference batches."""

from __future__ import annotations

import random

import torch
from data.base_dataset import get_transform
from PIL import Image
from util import util

from .dosc_unaligned_dataset import DoscUnalignedDataset


class TrscUnalignedDataset(DoscUnalignedDataset):
    """Preserve strict source labels and return K unlabeled target references."""

    def __init__(self, opt):
        super().__init__(opt)
        requested_references = int(getattr(opt, "trsc_num_references", 1))
        if requested_references < 1:
            raise ValueError("trsc_num_references must be at least one")
        self.reference_count = requested_references if opt.isTrain else 1
        if self.reference_count > self.B_size:
            raise ValueError(
                f"Requested {self.reference_count} unique target references, "
                f"but target train contains only {self.B_size} images"
            )
        self._target_index = {
            path: index for index, path in enumerate(self.B_paths)
        }

    def _additional_reference_indices(self, primary: int) -> list[int]:
        needed = self.reference_count - 1
        if needed <= 0:
            return []
        if self.opt.serial_batches:
            return [
                (primary + offset) % self.B_size
                for offset in range(1, self.reference_count)
            ]
        candidates = [index for index in range(self.B_size) if index != primary]
        return random.sample(candidates, needed)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        primary_index = self._target_index[item["B_paths"]]
        reference_paths = [item["B_paths"]]
        reference_paths.extend(
            self.B_paths[target_index]
            for target_index in self._additional_reference_indices(primary_index)
        )

        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(
            self.opt,
            load_size=self.opt.crop_size if is_finetuning else self.opt.load_size,
        )
        transform = get_transform(modified_opt)
        references = [item["B"]]
        for path in reference_paths[1:]:
            with Image.open(path) as image:
                references.append(transform(image.convert("RGB")))

        item["B_refs"] = torch.stack(references, dim=0)
        item["B_ref_paths"] = reference_paths
        return item
