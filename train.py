import argparse
import warnings
warnings.filterwarnings("ignore")

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import seed_everything
import torch

from vinafold.config import model_config
from vinafold.data.data_modules import VinaFoldDataModule
from vinafold.model.model import AlphaFold

from vinafold.utils.loss import AlphaFoldLoss
from vinafold.utils.tensor_utils import tensor_tree_map


class VinaFoldWrapper(pl.LightningModule):
    def __init__(self, config):
        super(VinaFoldWrapper, self).__init__()
        self.config = config

        self.model = AlphaFold(config)
        self.loss = AlphaFoldLoss(config.loss)

        self.save_hyperparameters()

    def forward(self, batch):
        return self.model(batch)

    def _log(self, loss_breakdown, train=True):
        phase = "train" if train else "val"
        for loss_name, indiv_loss in loss_breakdown.items():
            self.log(
                f"{phase}/{loss_name}", 
                indiv_loss, 
                prog_bar=(loss_name == 'loss'),
                on_step=train, on_epoch=(not train), logger=True, sync_dist=False,
            )

            if (train):
                self.log(
                    f"{phase}/{loss_name}_epoch",
                    indiv_loss,
                    on_step=False, on_epoch=True, logger=True, sync_dist=False,
                )

    def training_step(self, batch, batch_idx):
        batch.pop('gt_features', None)
        outputs = self(batch)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)
        loss, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        self._log(loss_breakdown)

        return loss

    def validation_step(self, batch, batch_idx):
        batch.pop('gt_features', None)
        outputs = self(batch)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)
        batch["use_clamped_fape"] = 0.

        _, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        self._log(loss_breakdown)

    def configure_optimizers(self, 
        learning_rate: float = 1e-3,
        eps: float = 1e-5,
    ) -> torch.optim.Adam:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            eps=eps
        )

def main(args):
    if(args.seed is not None):
        seed_everything(args.seed, workers=True) 

    config = model_config()
    model_module = VinaFoldWrapper(config)

    data_module = VinaFoldDataModule(
        config=config.data,
        batch_seed=args.seed,
        train_data_dir=args.train_data_dir,
        train_alignment_dir=args.train_alignment_dir
    )
    data_module.prepare_data()
    data_module.setup()

    if args.gpus is not None and args.gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = "auto"
 
    trainer = pl.Trainer(
        strategy=strategy,
        num_nodes=1,
        precision="bf16",
        max_epochs=5
    )

    trainer.fit(
        model_module,
        datamodule=data_module,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "train_data_dir", type=str,
        help="Directory containing training mmCIF files"
    )
    parser.add_argument(
        "train_alignment_dir", type=str,
        help="Directory containing precomputed training alignments"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help='For determining optimal strategy and effective batch size.'
    )

    args = parser.parse_args()
    main(args)
