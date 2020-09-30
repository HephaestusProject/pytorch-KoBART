import copy
import multiprocessing
import os
import random
import time
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm
from transformers import BartTokenizer


def get_masked_encode(encode, mask_ratio, mask_token_id):
    arange = np.arange(len(encode) - 1)
    np.random.shuffle(arange)
    mask_index = arange[: int(len(encode) * mask_ratio)]
    mask_index += 1
    masked_encode = copy.deepcopy(encode)
    masked_encode = np.stack(masked_encode)
    masked_encode[mask_index] = mask_token_id
    return masked_encode


def make_dataset(split_lines, tokenizer, repeat, mask_ratio, path):
    max_length = 0
    index = 0
    max_index = len(split_lines) * repeat

    dataset_path = os.path.join(path, f"dataset.h5")
    hf = h5py.File(dataset_path, "w")

    for line in tqdm.tqdm(split_lines):

        encode = tokenizer.encode(line)

        for _ in range(repeat):
            mask_encode = get_masked_encode(
                encode=encode,
                mask_ratio=mask_ratio,
                mask_token_id=tokenizer.mask_token_id,
            )
            hf.create_dataset(f"{index}/encode", data=encode)
            hf.create_dataset(f"{index}/mask_encode", data=mask_encode)
            index += 1

        max_length = max([max_length, len(encode)])

    for i in range(index):
        hf.create_dataset(f"{i}/max_length", data=max_length)
    hf.close()


class MakeBartMLMDataset:
    def __init__(self, tokenizer, source_io, mask_ratio, repeat, comment_dir):

        self.tokenizer = tokenizer
        self.mask_ratio = mask_ratio
        self.repeat = repeat

        lines = self.read_source(source_io=source_io)

        make_dataset(
            split_lines=lines,
            tokenizer=self.tokenizer,
            repeat=repeat,
            mask_ratio=mask_ratio,
            path=comment_dir,
        )

    def read_source(self, source_io):
        lines = []
        s = time.time()
        for line in source_io:
            line = line.replace("\n", "")
            lines.append(line)
            if len(lines) % 1000 == 0:
                print(f"read lines ... {len(lines)}, {time.time() - s}")
                s = time.time()

        print(f"read lines ... {len(lines)}, {time.time() - s}")
        return lines

    def get_split_index(self, total_length, n_proc, index):
        split = int(total_length / n_proc)
        start_index = index * split
        end_index = (index + 1) * split
        return start_index, end_index


def main(args):

    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path)

    proj_dir = Path()
    corpus_dir = proj_dir / "corpus"
    comment_dir = corpus_dir / "comment"
    source_path = comment_dir / "20190101_20200611_v2.txt"
    dataset_path = comment_dir / "dataset.h5py"

    source_io = open(source_path, "r", encoding="utf-8")

    make_dataset = MakeBartMLMDataset(
        tokenizer=tokenizer,
        source_io=source_io,
        mask_ratio=args.mask_ratio,
        repeat=args.repeat,
        comment_dir=comment_dir,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--repeat", default=1, type=int)
    parser.add_argument("--mask_ratio", default=0.15, type=float)
    parser.add_argument("--tokenizer_path", default="tokenizers", type=str)
    args = parser.parse_args()
    main(args)
