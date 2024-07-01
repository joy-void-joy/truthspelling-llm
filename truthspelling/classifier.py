# Does not work due to a bug in the way embedding.py saved the embeddings

import torch.utils.tensorboard
from collections import OrderedDict
import pathlib
import random
import shutil
from itertools import groupby

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

import numpy as np

import torch
import lightning.pytorch as pl
import lightning.pytorch.loggers
import torch.nn as nn
from torch.nn import functional as F

import torch.utils.data as tdata
import torchmetrics
import torchmetrics.classification
import pydantic
from torch.utils.data import Subset, ConcatDataset

import uuid

# Enter the path to your run here
data_dir = pathlib.Path("./data/scenarios/a2b7701778d64d8395823410aec38be2")
run_name = uuid.uuid4().hex


class EmbeddingDataset(tdata.IterableDataset):
    def __init__(self):
        self.embeddings = list(data_dir.glob("**/*.npy"))
        random.shuffle(self.embeddings)

    def __iter__(self):
        for emb in self.embeddings:
            honest = str(emb).endswith("honest.npy")

            yield (
                np.fromfile(emb, dtype=np.float64),
                honest,
            )


class Classifier(pl.LightningModule):
    class Constants(pydantic.BaseModel):
        batch_size: int
        learning_rate: float

    constants: Constants

    class Dataset(pydantic.BaseModel):
        train: tdata.Dataset | None
        val: tdata.Dataset | None
        test: tdata.Dataset | None

        class Config:
            arbitrary_types_allowed = True

    dataset: Dataset

    def __init__(
        self,
    ):
        super().__init__()
        self.save_hyperparameters()

        def get_constants() -> "Classifier.Constants":
            return Classifier.Constants(
                learning_rate=1e-4,
                batch_size=256,
            )

        self.constants = get_constants()

        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.f1 = torchmetrics.classification.BinaryF1Score()

        def get_model():
            def factory_linear(
                input_size,
                output_size=1,
                activation: type[nn.Module] = nn.Sigmoid,
                activation_kwargs={},
            ):
                return nn.Sequential(
                    OrderedDict(
                        {
                            "lin": nn.Linear(input_size, output_size),
                            "activation": activation(**activation_kwargs),
                        }
                    )
                )

            return nn.Sequential(
                OrderedDict(
                    {
                        "main": factory_linear(768, 1),
                    }
                )
            )

        self.model = get_model()

    ### DATA
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train, val, test = None, None, None
        full_dataset = EmbeddingDataset()
        train, val, test = full_dataset, full_dataset, full_dataset

        self.dataset = Classifier.Dataset(train=train, val=val, test=test)

    def dataloader(self, type, **kwargs):
        return tdata.DataLoader(
            getattr(self.dataset, type),
            batch_size=self.constants.batch_size,
            num_workers=24,
            **kwargs,
        )

    def train_dataloader(self):
        return self.dataloader("train")

    def val_dataloader(self):
        return self.dataloader("val")

    def test_dataloader(self):
        return self.dataloader("test")

    ### TRAINING
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        logits = self(x)
        loss = F.binary_cross_entropy(logits, y.unsqueeze(1))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        logits = self(x)

        loss = F.binary_cross_entropy(logits, y.unsqueeze(1))
        preds = logits
        self.accuracy(preds, y.unsqueeze(1))
        self.f1(preds, y.unsqueeze(1))

        self.log("val_f1", self.f1)
        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy)
        return {"loss": loss, "log": {"val_loss": loss, "val_acc": self.accuracy}}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.constants.learning_rate)

    ### TES
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


save_dir = pathlib.Path("./data") / "classifier" / run_name
log_dir = pathlib.Path("./data/lightning_logs") / "classifier" / run_name
log_dir.mkdir(exist_ok=True, parents=True)
save_dir.mkdir(exist_ok=True, parents=True)
(save_dir / "train.py").write_text(pathlib.Path(__file__).read_text())


def train(model, save_path, stopping_threshold=0.95):
    pl.seed_everything(
        random.randint(0, 100000),
    )

    model.setup()
    logger = pl.loggers.TensorBoardLogger("lightning_logs", version=f"{run_name}")

    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        every_n_epochs=20,
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=200,
        logger=logger,
        # deterministic=True
        callbacks=[model_checkpoint],
    )

    trainer.fit(model)

    trainer.save_checkpoint(save_path)


if __name__ == "__main__":
    model = Classifier()
    train(model, save_path=save_dir / "classifier.ckpt", stopping_threshold=0.9)
