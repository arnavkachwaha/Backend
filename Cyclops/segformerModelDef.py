import os
import torch
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint 
from torch.utils.data import DataLoader, IterableDataset
from transformers import SegformerForSemanticSegmentation
import torch
import torchmetrics
import torch.nn.functional as F


# Normalization transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CyclopsSegformerModule(pl.LightningModule):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            'nvidia/segformer-b0-finetuned-ade-512-512',
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        self.test_losses = []
        self.test_ious = []
        self.metrics = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes)

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images, labels=labels)
        loss = outputs.loss
        self.log('train_loss', loss)

        # Upsample logits to match the size of labels
        logits = outputs.logits
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.argmax(logits, dim=1)

        self.metrics(preds, labels)
        self.log('train_iou', self.metrics, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images, labels=labels)
        val_loss = outputs.loss
        self.log('val_loss', val_loss)

        # Upsample logits to match the size of labels
        logits = outputs.logits
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.argmax(logits, dim=1)

        self.metrics(preds, labels)
        self.log('val_iou', self.metrics, on_step=False, on_epoch=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images, labels=labels)
        loss = outputs.loss
        self.log('test_loss', loss)

        # Upsample logits to match the size of labels
        logits = outputs.logits
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.argmax(logits, dim=1)

        self.metrics(preds, labels)
        self.log('test_iou', self.metrics, on_step=False, on_epoch=True)

        self.test_losses.append(loss.detach())
        iou = self.metrics.compute()
        self.test_ious.append(iou)
        return {'test_loss': loss, 'iou': iou}

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_losses).mean()
        avg_iou = torch.stack(self.test_ious).mean()
        self.log('avg_test_loss', avg_loss)
        self.log('avg_iou_score', avg_iou)
        self.test_losses.clear()
        self.test_ious.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        return optimizer