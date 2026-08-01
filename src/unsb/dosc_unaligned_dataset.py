"""UNSB unaligned dataset overlay with labeled source-domain metadata.

Domain A must be the labeled source domain. Domain B is the unlabeled target
domain and no target label file is accepted by this dataset.
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path

import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
from util import util


class DoscUnalignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            "--dosc_source_manifest",
            type=str,
            default="",
            help="CSV mapping trainA relative_path to source label and case_id",
        )
        parser.add_argument("--dosc_manifest_path_col", type=str, default="relative_path")
        parser.add_argument("--dosc_manifest_label_col", type=str, default="label")
        parser.add_argument("--dosc_manifest_case_col", type=str, default="case_id")
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")
        if (
            opt.phase == "test"
            and not os.path.exists(self.dir_A)
            and os.path.exists(os.path.join(opt.dataroot, "valA"))
        ):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        if self.A_size == 0 or self.B_size == 0:
            raise ValueError(
                f"Both domains must contain images, got A={self.A_size}, B={self.B_size}"
            )

        self.source_metadata: dict[str, tuple[int, str]] = {}
        manifest_value = str(opt.dosc_source_manifest).strip()
        if opt.isTrain:
            manifest_path = (
                Path(manifest_value)
                if manifest_value
                else Path(opt.dataroot) / "trainA_manifest.csv"
            )
            self.source_metadata = self._load_source_manifest(manifest_path)
            missing = [
                path
                for path in self.A_paths
                if self._metadata_key(path) not in self.source_metadata
            ]
            if missing:
                preview = ", ".join(missing[:3])
                raise KeyError(
                    f"{len(missing)} trainA images are absent from {manifest_path}: {preview}"
                )

    def _normalize_relative_path(self, value: str) -> str:
        return Path(str(value)).as_posix().lstrip("./")

    def _metadata_key(self, image_path: str) -> str:
        relative = os.path.relpath(image_path, self.opt.dataroot)
        return self._normalize_relative_path(relative)

    def _load_source_manifest(self, manifest_path: Path) -> dict[str, tuple[int, str]]:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Source manifest not found: {manifest_path}")
        path_column = self.opt.dosc_manifest_path_col
        label_column = self.opt.dosc_manifest_label_col
        case_column = self.opt.dosc_manifest_case_col
        mapping: dict[str, tuple[int, str]] = {}
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {path_column, label_column}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"{manifest_path} must contain columns {sorted(required)}; "
                    f"found {reader.fieldnames}"
                )
            for row_number, row in enumerate(reader, start=2):
                key = self._normalize_relative_path(row[path_column])
                label = int(row[label_column])
                if label < 0 or label >= int(self.opt.dosc_num_classes):
                    raise ValueError(
                        f"Invalid source label {label} at {manifest_path}:{row_number}"
                    )
                case_id = str(row.get(case_column, "")).strip() or Path(key).stem
                value = (label, case_id)
                if key in mapping and mapping[key] != value:
                    raise ValueError(f"Conflicting manifest rows for {key}")
                mapping[key] = value
        if not mapping:
            raise ValueError(f"Source manifest has no records: {manifest_path}")
        return mapping

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_image = Image.open(A_path).convert("RGB")
        B_image = Image.open(B_path).convert("RGB")
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(
            self.opt,
            load_size=self.opt.crop_size if is_finetuning else self.opt.load_size,
        )
        transform = get_transform(modified_opt)
        A = transform(A_image)
        B = transform(B_image)

        source_label = -1
        source_case_id = Path(A_path).stem
        if self.opt.isTrain:
            source_label, source_case_id = self.source_metadata[self._metadata_key(A_path)]

        return {
            "A": A,
            "B": B,
            "A_paths": A_path,
            "B_paths": B_path,
            "A_label": torch.tensor(source_label, dtype=torch.long),
            "A_case_id": source_case_id,
        }

    def __len__(self):
        return max(self.A_size, self.B_size)
