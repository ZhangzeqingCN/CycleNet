import torch
import torch.nn.functional as F
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import optim

from args import Args
from data_provider.data_factory import data_provider
from exp.build_model import build_model
from utils.settings import get_setting
from utils.tools import adjust_learning_rate


class WrapperModule(LightningModule):

    def __init__(self, args: Args):
        super().__init__()
        self.args = args
        self.model = build_model(args)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def base_step(self, batch, i) -> STEP_OUTPUT:
        batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, batch_rs = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        if any(substr in self.args.model for substr in {'MFRS'}):
            outputs = self.model(batch_x, batch_rs.float())
        elif any(substr in self.args.model for substr in {'Cycle'}):
            outputs = self.model(batch_x, batch_cycle)
        elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF'}):
            outputs = self.model(batch_x)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
        loss = F.mse_loss(outputs, batch_y)
        return loss

    def training_step(self, batch, i) -> STEP_OUTPUT:
        loss = self.base_step(batch, i)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def on_train_epoch_end(self) -> None:
        adjust_learning_rate(self.optimizers().optimizer, None, self.current_epoch + 1, self.args, printout=False)

    def validation_step(self, batch, i) -> STEP_OUTPUT:
        loss = self.base_step(batch, i)
        self.log("vali_loss", loss, prog_bar=True, on_epoch=True)

    def test_step(self, batch, i) -> STEP_OUTPUT:
        loss = self.base_step(batch, i)
        self.log("test_loss", loss, on_epoch=True)


class Exp_Lightning:

    def __init__(self, args: Args):
        if torch.cuda.is_available():
            devices = -1
        else:
            devices = 1
        self.args = args
        self.wrapper = WrapperModule(args)
        setting = get_setting(args)
        self.trainer = Trainer(
            devices=devices,
            max_epochs=args.train_epochs, num_sanity_val_steps=0, precision=32,
            callbacks=[
                EarlyStopping(
                    patience=args.patience,
                    monitor="vali_loss",
                    min_delta=0,
                    verbose=False,
                ),
                ModelCheckpoint(
                    save_last=True,
                    monitor="vali_loss"
                ),
            ],
            logger=[
                # CSVLogger("logs", name=setting),
                WandbLogger(project="time-series", version=setting, log_model=False, offline=True),
            ]
        )

    def train(self, setting):
        train_data, train_loader = data_provider(self.args, flag='train')
        vali_data, vali_loader = data_provider(self.args, flag='val')
        self.trainer.fit(model=self.wrapper, train_dataloaders=train_loader, val_dataloaders=vali_loader)

    def test(self, setting, test=0):
        test_data, test_loader = data_provider(self.args, flag='test')
        self.trainer.test(model=self.wrapper, dataloaders=test_loader)
