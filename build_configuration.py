import random
from argparse import ArgumentParser
from pathlib import Path

import tqdm
from transformers import BartConfig, BartModel, BartTokenizer


def main(args):

    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path)
    bart_config = BartConfig()
    bart_config.vocab_size = len(tokenizer)
    bart_config.eos_token_id = tokenizer.eos_token_id
    bart_config.bos_token_id = tokenizer.bos_token_id
    bart_config.pad_token_id = tokenizer.pad_token_id

    bart_config.save_pretrained(args.config_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tokenizer_path", default="tokenizers", type=str)
    parser.add_argument("--config_path", default="kobart", type=str)
    args = parser.parse_args()
    main(args)
