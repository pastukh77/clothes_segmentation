from module import DataModule, PlSeg
from pytorch_lightning import Trainer
from config import Config
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

if __name__ == "__main__":
    checkpoint = ModelCheckpoint(
            monitor="val_loss",
            dirpath='weights',
            filename="model-{epoch:02d}-{val_loss:.2f}",
            mode="min",
        )

    data_module = DataModule(Config)

    seg = PlSeg(Config.MODEL, Config.ENCODER, Config.ENCODER_WEIGHTS, num_classes=4, num_channels=Config.CHANNELS)

    trainer = Trainer(max_epochs=30,
                        callbacks=[checkpoint],
                        check_val_every_n_epoch=1,
                        val_check_interval=0.2,
                        auto_scale_batch_size=True)

    trainer.fit(seg, data_module)