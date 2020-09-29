import random
from argparse import ArgumentParser
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer


def main(args):
    # set the corpus
    random.seed(42)
    proj_dir = Path()
    tokenizers_dir = proj_dir / "tokenizers"

    if not tokenizers_dir.exists():
        tokenizers_dir.mkdir(parents=True)

    corpus_dir = proj_dir / "corpus"
    comment_dir = corpus_dir / "comment"
    source_path = comment_dir / "20190101_20200611_v2.txt"
    sample_path = comment_dir / "sample.txt"

    # sampling source
    source_io = open(source_path, mode="r", encoding="utf-8")
    sample_io = open(sample_path, mode="w", encoding="utf-8")

    for line in source_io:
        if random.random() > (1 - args.sample_rate):
            sample_io.write(line)
    else:
        sample_io.close()
        source_io.close()

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer(add_prefix_space=False)

    # Customize training
    tokenizer.train(
        files=str(sample_path),
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq,
        show_progress=True,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"],
    )
    tokenizer.save_model(directory=str(tokenizers_dir))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sample_rate", default=0.1, type=float)
    parser.add_argument("--vocab_size", default=30000, type=int)
    parser.add_argument("--min_freq", default=5, type=int)
    args = parser.parse_args()
    main(args)
