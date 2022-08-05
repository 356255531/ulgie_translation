import sys
import os
import argparse
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision import transforms

sys.path.append(os.path.join(os.getcwd()))

from ulgie.data import SimpleDataModule
from ulgie.method import SlotAttentionMethod
from ulgie.model import SlotAttentionModel
from ulgie.params import *
from ulgie.utils import ImageLogCallback
from ulgie.utils import rescale


def main(args):
    params = SlotAttentionParams()

    if params.is_verbose:
        print(f"INFO: limiting the dataset to only images with `num_slots - 1` ({params.num_slots - 1}) objects.")
        if params.num_train_images:
            print(f"INFO: restricting the train dataset size to `num_train_images`: {params.num_train_images}")
        if params.num_val_images:
            print(f"INFO: restricting the validation dataset size to `num_val_images`: {params.num_val_images}")

    simple_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(rescale),  # rescale between -1 and 1
            transforms.Resize(params.resolution),
        ]
    )

    datamodule = SimpleDataModule(
        data_root=params.data_root,
        max_n_objects=1,
        train_batch_size=params.batch_size,
        val_batch_size=params.val_batch_size,
        transforms=simple_transforms,
        num_train_images=params.num_train_images,
        num_val_images=params.num_val_images,
        num_workers=params.num_workers,
    )

    print(f"Training set size (images must have {params.num_slots - 1} objects):", len(datamodule.train_dataset))

    model = SlotAttentionModel(
        resolution=params.resolution,
        num_slots=params.num_slots,
        num_iterations=params.num_iterations,
        empty_cache=params.empty_cache,
    )

    method = SlotAttentionMethod(model=model, datamodule=datamodule, params=params)

    logger_name = f"{args.dataset}_transform"
    logger = pl_loggers.WandbLogger(project="ulgie", name=logger_name)

    trainer = Trainer(
        logger=logger if params.is_logger_enabled else False,
        accelerator="ddp" if len(params.gpus) > 1 else None,
        num_sanity_val_steps=params.num_sanity_val_steps,
        gpus=params.gpus,
        max_epochs=params.max_epochs,
        log_every_n_steps=50,
        callbacks=[LearningRateMonitor("step"), ImageLogCallback(),] if params.is_logger_enabled else [],
    )
    trainer.fit(method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, default='simple_color_shape', help='dataset to run')
    args = parser.parse_args()
    main(args)
