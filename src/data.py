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

    def __init__(self, dataset_path, tokenizer):

        self.hf = h5py.File(dataset_path, 'r')
        self.max_length = np.array(self.hf.get('max_length'))
        self.data_num = np.array(self.hf.get('data_num'))
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        encode = np.array(self.hf.get(f'{idx}/encode'))
        masked_encode = np.array(self.hf.get(f'{idx}/mask_encode'))

        print(encode)
        print(masked_encode)
        padded_encode, padded_attention_mask = self.padding(encode)
        padded_masked_encode, _ = self.padding(masked_encode)

        padded_encode = torch.as_tensor(padded_encode, dtype=torch.long)
        padded_attention_mask = torch.as_tensor(padded_attention_mask, dtype=torch.bool)
        padded_masked_encode = torch.as_tensor(padded_masked_encode, dtype=torch.long)
        return padded_masked_encode, padded_attention_mask, padded_encode

    def padding(self, encode):
        encode = list(encode)
        diff_length = self.max_length - len(encode)
        attention_mask = [1] * len(encode) + [0] * diff_length
        
        append_encode = [self.tokenizer.pad_token_id] * diff_length
        encode.extend(append_encode)
        return encode, attention_mask

    def __len__(self):
        return self.data_num

class BartDataModule(pl.LightningDataModule):

    def __init__(self, dataset_path, batch_size, num_workers, tokenizer):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        BartDataset(self.dataset_path, self.tokenizer)

    def train_dataloader(self):
        dataset = BartDataset(
                self.dataset_path, self.tokenizer)
        dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False)
        return dataloader

def main(args):

    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path)

    proj_dir = Path()
    corpus_dir = proj_dir / "corpus"
    comment_dir = corpus_dir / "comment"

    h5py_files = comment_dir / "dataset.h5py"

    dm = BartDataModule(h5py_files, 1, 1, tokenizer)
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
