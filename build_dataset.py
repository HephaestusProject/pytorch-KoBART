import copy
import json
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

def get_masking_index(line, tokenizer, mask_ratio):
    encode = tokenizer.encode(line)
    arange = np.arange(len(encode)-2)
    np.random.shuffle(arange)
    masked_index = arange[:int(len(encode) * mask_ratio)]
    masked_index += 1
    masked_index = [int(m) for m in masked_index]
    return masked_index

def make_masked_index(source_io, tokenizer, mask_ratio):

    N = 10
    num_cpu = 8
    s = time.time()
    lines = []
    masked_indexes = []
    for step, line in enumerate(source_io):
        line = line.replace('\n', '')
        lines.append([
            get_masking_index, line,
            tokenizer, mask_ratio])

        if len(lines) == N:
            pool = multiprocessing.Pool(num_cpu)
            masked_index = pool.map(map_wrapper, lines)
            masked_indexes.extend(masked_index)
            print(f'make masked index ... {step} ... {time.time() - s}')
            lines = []
            s = time.time()

    pool = multiprocessing.Pool(num_cpu)
    masked_index = pool.map(map_wrapper, lines)
    masked_indexes.extend(masked_index)

    print(f'make masked index ... {step} ... {time.time() - s}')

    return masked_indexes

def main(args):

    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path)

    proj_dir = Path()
    corpus_dir = proj_dir / "corpus"
    comment_dir = corpus_dir / "comment"
    source_path = comment_dir / args.corpus
    dataset_path = comment_dir / args.mask_path

    source_io = open(source_path, "r", encoding="utf-8")
    masked_index = make_masked_index(source_io=source_io,
                                     tokenizer=tokenizer,
                                     mask_ratio=args.mask_ratio)
    with open(dataset_path, 'w') as f:
        json.dump(masked_index, f)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mask_ratio", default=0.15, type=float)

    # parser.add_argument("--corpus", default="20190101_20200611_v2.txt", type=str)
    parser.add_argument("--corpus", default="test.txt", type=str)
    
    parser.add_argument("--mask_path", default="dataset.json", type=str)
    parser.add_argument("--tokenizer_path", default="tokenizers", type=str)
    args = parser.parse_args()
    main(args)
