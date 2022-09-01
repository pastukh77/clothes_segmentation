
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torchmetrics as tm

from data import FashionDataset

import albumentations as A

from torch.utils.data import random_split, DataLoader


class PlSeg(pl.LightningModule):
    def __init__(self, model, encoder, encoder_weights, num_classes, num_channels, batch_size=32):
        super().__init__()
        self.num_classes = num_classes
        self.model = smp.create_model(model, encoder, encoder_weights, num_channels, self.num_classes + 1)
        self.batch_size = batch_size
        params = smp.encoders.get_preprocessing_params(encoder)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.loss_fn = smp.losses.JaccardLoss(mode="multiclass", from_logits=True)

        metrics = tm.MetricCollection([
            tm.Accuracy(num_classes=self.num_classes + 1, mdmc_reduce="global"), 
            tm.Dice(num_classes=self.num_classes + 1, mdmc_reduce="global"), 
            tm.Recall(self.num_classes + 1, mdmc_reduce="global"), 
            tm.F1Score(self.num_classes + 1, mdmc_reduce="global")
            ])
        
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')


    def forward(self, image):
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, mode):
        img, mask = batch
        output = self(img)

        result = torch.argmax(output, dim=1)

        if mode == "train":
            metrics = self.train_metrics(mask, result)
        elif mode == "val":
            metrics = self.valid_metrics(mask, result)

        loss = self.loss_fn(output, mask.long())
        self.log(f"{mode}_loss", loss)
        self.log_dict(metrics)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super(DataModule, self).__init__()
        self.transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussNoise()])
        self.batch_size = config.BATCH_SIZE
        fashionDataset = FashionDataset(config.PATH, config.IMAGE_SIZE, transforms=self.transforms)
        train_size = int(len(fashionDataset) * 0.8)
        self.train_ds, self.val_ds = random_split(fashionDataset, [train_size, len(fashionDataset) - train_size])

    def train_dataloader(self):
        return DataLoader(self.train_ds, pin_memory=True, shuffle=True, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_ds, pin_memory=True, batch_size=self.batch_size, num_workers=8)
