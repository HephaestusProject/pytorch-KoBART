"""
    This script was made by Nick at 19/07/20.
    To implement code for training your model.
"""
import pytorch_lightning
import argparse
import torch

import pytorch_lightning as pl

from src.module import BartModule
from src.data import BartDataModule
from pathlib import Path
from transformers import BartTokenizer

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers

pytorch_lightning.seed_everything(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_path', default='tokenizers', type=str)
    parser.add_argument('--corpus', default='test.txt', type=str)
    parser.add_argument('--mask_path', default='dataset.json', type=str)
    parser.add_argument('--config_path', default='kobart', type=str)
    parser.add_argument('--logger', default='kobart', type=str)
    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--warmup_epochs', default=3, type=int)
    args = parser.parse_args()

    proj_dir = Path()
    corpus_dir = proj_dir / "corpus"
    comment_dir = corpus_dir / "comment"
    source_path = comment_dir / args.corpus
    mask_path = comment_dir / args.mask_path

    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path)
    dm = BartDataModule(
            source_path=source_path,
            mask_path=mask_path,
            tokenizer=tokenizer,
            batch_size=2,
            num_workers=1)
    dm.setup()
    train_dataloader = dm.train_dataloader()    
    
    checkpoint_callback = ModelCheckpoint(
            save_top_k=-1, verbose=True)
    logger = loggers.TensorBoardLogger(args.logger)
    model = BartModule(
            config=args.config_path,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            warmup_epochs=args.warmup_epochs)

    device_count = torch.cuda.device_count()
    trainer = pl.Trainer(
            # gpus=device_count,
            # distributed_backend='ddp',
            max_epochs=args.max_epochs,
            checkpoint_callback=checkpoint_callback,
            logger=logger)

    trainer.fit(model, train_dataloader)

if __name__ == '__main__':
    main()
