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

def map_wrapper(x):
    return x[0](*(x[1:]))

def read_txt_file(source_io, repeat, dataset_path, tokenizer, mask_ratio):
    input_args = []
    t = time.time()
    s = time.time()
    index = 0
    for line in source_io:
        line = line.replace("\n", "")
        for _ in range(repeat):
            input_args.append([line, index,
                               dataset_path, tokenizer, mask_ratio])
            index += 1
        if len(input_args) % 1000 == 0:
            print(f"read lines ... {len(input_args)}, {time.time() - s}")
            s = time.time()
    print(f"read all lines ... {len(input_args)}, {time.time() - t}")
    return input_args

def masking_dataset(line, index, dataset_path, tokenizer, mask_ratio):
    token = tokenizer.encode(line)
    arange = np.arange(len(token) - 2)
    np.random.shuffle(arange)
    mask_index = arange[:int(len(arange) * mask_ratio)]
    mask_index += 1
    masked_token = copy.deepcopy(token)
    for mask in mask_index:
        masked_token[mask] = tokenizer.mask_token_id

    return token, masked_token, index

def write_dataset(dataset_path, lines):

    hf = h5py.File(dataset_path, 'w')
    max_length = 0
    s = time.time()
    t = time.time()
    for line in lines:
        token, masked_token, index = masking_dataset(*line)
        hf.create_dataset(f'{index}/encode', data=token)
        hf.create_dataset(f'{index}/mask_encode', data=masked_token)

        if index % 1000 == 0:
            print(f"write dataset into h5 format...{index} / {len(lines)}, {time.time() - s}")
            s = time.time()

        if max_length <= len(token):
            max_length = len(token)

    hf.create_dataset('data_num', data=len(lines))
    hf.create_dataset('max_length', data=max_length)

    print(f"write all dataset into h5 format...{index} / {len(lines)}, {time.time() - t}")

def read_write_data(source_io, repeat, dataset_path, tokenizer, mask_ratio):
    
    hf = h5py.File(dataset_path, 'w')
    t = time.time()
    s = time.time()
    max_length = 0
    data_num = []
    for step, line in enumerate(source_io):
        token, masked_token, index = masking_dataset(
                line=line, index=step, dataset_path=dataset_path,
                tokenizer=tokenizer, mask_ratio=mask_ratio)

        if max_length <= len(token):
            max_length = len(token)
        
        hf.create_dataset(f'{index}/encode', data=token)
        hf.create_dataset(f'{index}/mask_encode', data=masked_token)
        data_num.append(None)

        if len(data_num) % 1000 == 0:
            print(f"write dataset into h5 format...{len(data_num)}, {time.time() - s}")
            s = time.time()

        if len(data_num) == 3000:
            break

    hf.create_dataset('data_num', data=len(data_num))
    hf.create_dataset('max_length', data=max_length)
    print(f"write dataset into h5 format...{len(data_num)}, {time.time() - t}")

def main(args):

    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path)

    proj_dir = Path()
    corpus_dir = proj_dir / "corpus"
    comment_dir = corpus_dir / "comment"
    source_path = comment_dir / "20190101_20200611_v2.txt"
    dataset_path = comment_dir / "dataset.h5py"

    source_io = open(source_path, "r", encoding="utf-8")
    read_write_data(
            source_io=source_io,
            repeat=args.repeat,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            mask_ratio=args.mask_ratio)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--repeat", default=2, type=int)
    parser.add_argument("--mask_ratio", default=0.15, type=float)
    parser.add_argument("--tokenizer_path", default="tokenizers", type=str)
    args = parser.parse_args()
    main(args)
