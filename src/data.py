"""
    This script was made by Nick at 19/07/20.
    To implement code for data pipeline. (e.g. custom class subclassing torch.utils.data.Dataset)
"""

import copy
import os
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import tqdm
from transformers import BartTokenizer


class BartDataset(torch.utils.data.Dataset):
    def __init__(self, h5py_path, tokenizer):
        super().__init__()

        self.h5py = h5py.File(h5py_path, "r")
        self.tokenizer = tokenizer
        self.max_length = np.array(self.h5py.get(f"0/max_length"))

    def __getitem__(self, idx):
        encode = np.array(self.h5py.get(f"{idx}/encode"))
        masked_encode = np.array(self.h5py.get(f"{idx}/mask_encode"))

        padded_encode, padded_attention_mask = self.padding(
            encode=encode, max_length=self.max_length
        )
        padded_masked_encode, _ = self.padding(
            encode=masked_encode, max_length=self.max_length
        )

        padded_encode = torch.as_tensor(padded_encode, dtype=torch.long)
        padded_attention_mask = torch.as_tensor(padded_attention_mask, dtype=torch.bool)
        padded_masked_encode = torch.as_tensor(padded_masked_encode, dtype=torch.long)
        return padded_masked_encode, padded_attention_mask, padded_encode

    def padding(self, encode, max_length):

        encode = list(encode)

        diff_length = max_length - len(encode)
        attention_mask = [True] * len(encode) + [False] * diff_length

        append_encode = [self.tokenizer.pad_token_id] * diff_length
        encode.extend(append_encode)

        return encode, attention_mask

    def __len__(self):
        return len(self.h5py)


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, h5py_paths, tokenizer):

        self.datasets = [BartDataset(h5py_path, tokenizer) for h5py_path in h5py_paths]
        self.max_length = max([dataset.max_length for dataset in self.datasets])

        for dataset in self.datasets:
            dataset.max_length = self.max_length

        self.concat_dataset = []
        for dataset in self.datasets:
            for i in tqdm.tqdm(np.arange(len(dataset))):
                self.concat_dataset.append(dataset[i])

    def __getitem__(self, idx):
        return self.concat_dataset[idx]

    def __len__(self):
        return len(self.concat_dataset)


class BartDataModule(pl.LightningDataModule):
    def __init__(self, h5py_paths, batch_size, num_workers, tokenizer):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.h5py_paths = h5py_paths
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        ConcatDataset(htpy_paths=self.h5py_paths, tokenizer=self.tokenizer)

    def train_dataloader(self):
        dataset = ConcatDataset(h5py_paths=self.h5py_paths, tokenizer=self.tokenizer)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        return dataloader


def main(args):

    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path)

    proj_dir = Path()
    corpus_dir = proj_dir / "corpus"
    comment_dir = corpus_dir / "comment"

    h5py_files = []
    for roots, dirs, files in os.walk(comment_dir):
        for fname in files:
            _, ext = os.path.splitext(fname)
            if ext == ".h5":
                h5py_files.append(os.path.join(roots, fname))

    dm = BartDataModule(
        h5py_paths=h5py_files, batch_size=1, num_workers=16, tokenizer=tokenizer
    )

    train_loader = dm.train_dataloader()
    for input_ids, attention_mask, labels in train_loader:

        input_ids = input_ids[0].detach().cpu().numpy()
        labels = labels[0].detach().cpu().numpy()

        print(tokenizer.decode(input_ids))
        print(tokenizer.decode(labels))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--tokenizer_path", default="tokenizers", type=str)
    args = parser.parse_args()
    main(args)
