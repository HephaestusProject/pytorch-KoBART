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
import json

from transformers import BartTokenizer

class BartDataset(torch.utils.data.IterableDataset):

    def __init__(self, source_path, mask_path, tokenizer):

        self.source_path = source_path
        self.mask_path = mask_path
        self.tokenizer = tokenizer

    def line_mapper(self, line, mask):
        line = line.replace('\n', '')
        encode = self.tokenizer.encode(line)
        masked_encode = copy.deepcopy(encode)
        for m in mask:
            masked_encode[m] = self.tokenizer.mask_token_id

        attention_mask = [1] * len(masked_encode)

        masked_encode = torch.as_tensor(masked_encode, dtype=torch.long)
        encode = torch.as_tensor(encode, dtype=torch.long)
        attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)

        return masked_encode, attention_mask, encode

    def __iter__(self):
        source_iter = open(self.source_path, 'r', encoding='utf-8')

        mask_data = open(self.mask_path, 'r')
        mask_iter = json.load(mask_data)

        mapped_source_iter = map(self.line_mapper, source_iter, mask_iter)

        return mapped_source_iter

class BartDataModule(pl.LightningDataModule):

    def __init__(self, source_path, mask_path, tokenizer, batch_size, num_workers):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.source_path = source_path
        self.mask_path = mask_path
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        BartDataset(
                source_path=self.source_path,
                mask_path=self.mask_path,
                tokenizer=self.tokenizer)

    def collate_fn(self, data):
        
        def merge(sequence, pad_num):
            lengths = [len(seq) for seq in sequence]
            padded_seqs = torch.ones(len(sequence), max(lengths)).long()
            padded_seqs = padded_seqs * pad_num    
            for i, seq in enumerate(sequence):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs

        masked_encode, attention_mask, encode = zip(*data)

        masked_encode = merge(masked_encode, self.tokenizer.pad_token_id)
        encode = merge(encode, self.tokenizer.pad_token_id)
        attention_mask = merge(attention_mask, pad_num=0)

        return masked_encode, attention_mask, encode

    def train_dataloader(self):
        dataset = BartDataset(
                source_path=self.source_path,
                mask_path=self.mask_path,
                tokenizer=self.tokenizer)

        dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                drop_last=True)

        return dataloader

def main(args):

    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path)

    proj_dir = Path()
    corpus_dir = proj_dir / "corpus"
    comment_dir = corpus_dir / "comment"

    source_path = comment_dir / args.corpus
    mask_path = comment_dir / args.mask_path

    dm = BartDataModule(
            source_path=source_path,
            mask_path=mask_path,
            tokenizer=tokenizer,
            batch_size=3,
            num_workers=1)

    data_loader = dm.train_dataloader()

    for masked_encode, attention_mask, encode in data_loader:
        masked_encode = masked_encode.detach().cpu().numpy()
        encode = encode.detach().cpu().numpy()

        for m, e in zip(masked_encode, encode):
            print(tokenizer.decode(m))
            print(tokenizer.decode(e))
            print(attention_mask)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--tokenizer_path", default="tokenizers", type=str)
    # parser.add_argument("--corpus", default="20190101_20200611_v2.txt", type=str)
    parser.add_argument("--corpus", default="test.txt", type=str)
    parser.add_argument("--mask_path", default="dataset.json", type=str)

    args = parser.parse_args()
    main(args)

