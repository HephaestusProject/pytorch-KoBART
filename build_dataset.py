import copy
import collections
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

Returns = collections.namedtuple('Returns', 
        ['target_token', 'masked_token', 'length', 'index'])

def map_wrapper(x):
    return x[0](*(x[1:]))

def get_single_dataset(line, tokenizer, mask_ratio, index):
    token = tokenizer.encode(line)
    arange = np.arange(len(token) - 2)
    np.random.shuffle(arange)
    mask_index = arange[:int(len(arange) * mask_ratio)]
    mask_index += 1
    masked_token = copy.deepcopy(token)
    for mask in mask_index:
        masked_token[mask] = tokenizer.mask_token_id
    data = Returns(
            target_token=token,
            masked_token=masked_token,
            length=len(token),
            index=index)
    if index % 1000 == 0:
        print(f'masking token ... {index}')
    return data

def read_txt_file(source_io, tokenizer,
                  mask_ratio, repeat):
    input_args = []
    t = time.time()
    s = time.time()
    index = 0
    for line in source_io:
        line = line.replace("\n", "")
        for _ in range(repeat):
            input_args.append([
                get_single_dataset,
                line, tokenizer,
                mask_ratio, index])
            index += 1
        if len(input_args) % 1000 == 0:
            print(f"read lines ... {len(input_args)}, {time.time() - s}")
            s = time.time()
    print(f"read all lines ... {len(input_args)}, {time.time() - t}")
    return input_args

def write_file(dataset_path, datasets):

    hf = h5py.File(dataset_path, 'w')
    max_length = 0
    s = time.time()
    for step, dataset in enumerate(datasets):
        if max_length <= dataset.length:
            max_length = dataset.length
        hf.create_dataset(f'{dataset.index}/encode', data=dataset.target_token)
        hf.create_dataset(f'{dataset.index}/mask_encode', data=dataset.masked_token)

        if step % 1000 == 0:
            print(f"write dataset into h5 format ... {step} / {len(datasets)}, {time.time() - s}")
            s = time.time()

    hf.create_dataset('data_num', data=len(datasets))
    hf.create_dataset('max_length', data=max_length)

    print(f"write dataset all into h5 format")

def main(args):

    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path)

    proj_dir = Path()
    corpus_dir = proj_dir / "corpus"
    comment_dir = corpus_dir / "comment"
    source_path = comment_dir / "20190101_20200611_v2.txt"
    dataset_path = comment_dir / "dataset.h5py"

    source_io = open(source_path, "r", encoding="utf-8")
    input_args = read_txt_file(source_io, tokenizer,
                               args.mask_ratio, args.repeat)

    pool = multiprocessing.Pool(args.num_cpu)
    datasets = pool.map(map_wrapper, input_args)

    write_file(dataset_path, datasets)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--repeat", default=10, type=int)
    parser.add_argument("--num_cpu", default=8, type=int)
    parser.add_argument("--mask_ratio", default=0.15, type=float)
    parser.add_argument("--tokenizer_path", default="tokenizers", type=str)
    args = parser.parse_args()
    main(args)
