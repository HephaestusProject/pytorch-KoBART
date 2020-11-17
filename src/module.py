import pytorch_lightning as pl
import torch
import torchvision

from transformers import BartForConditionalGeneration
from transformers import BartConfig

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.optimizers.lars_scheduling import LARSWrapper

class BartModule(pl.LightningModule):

    def __init__(self, config, learning_rate,
                 weight_decay, max_epochs, warmup_epochs):
        super(BartModule, self).__init__()

        self.save_hyperparameters()
        self.config = BartConfig.from_pretrained(config)
        self.model = BartForConditionalGeneration(
                config=self.config)

        self.train_step = 0

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True, output_hidden_states=True)
        return output

    def shared_step(self, batch):

        input_ids, attention_mask, labels = batch
        output = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels)
        loss = output.loss
        
        pred = torch.argmax(output.logits, axis=2)
        pred = pred.eq(labels).view(-1).to(dtype=torch.float)
        pred = pred.mean()
        return loss, pred

    def training_step(self, batch, batch_idx):

        self.train_step += 1

        loss, pred = self.shared_step(batch)
        
        self.logger.experiment.add_scalar('data/train_loss', loss, self.train_step)
        self.logger.experiment.add_scalar('data/train_pred', pred, self.train_step)
        self.logger.experiment.add_scalar('data/lr', self.optimizers[0].param_groups[0]['lr'], self.train_step)

        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay)

        optimizer = LARSWrapper(optimizer)

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs
        )

        self.optimizers = [optimizer]

        return self.optimizers, [scheduler]

