import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from relationship.data import RelationshipExtractionDataset
from relationship.korre import KingKorre
from torch.utils.data import DataLoader
import lightning as l

from lightning.pytorch.loggers import WandbLogger
from fire import Fire
import os
import torch
from datetime import datetime as dt


def main(
    project: str = "KingKorre",
    gpu=0,
    log_model: str = "all",
    pooling_mode: str = "cls",
    rel2id_path: str = "./gpt_relationships_only_person.json",
    max_epochs: int = 3,
    train_json_path: str = "sample_data/태조7월.json",
    valid_json_path: str = "sample_data/태조7월.json",
    batch_size: int = 16,
    max_len: int = 512,
    save_dir: str = "trained_models",
    train_val_split: (
        float | None
    ) = None,  # if train_val_split is not None, then use train_json only and split it into train and valid with the ratio of train_val_split.
):
    time_now = dt.now().strftime("%Y%m%d%H%M%S")
    run_name = f"{project}_batch{batch_size}_pooling_{pooling_mode}_maxepochs_{max_epochs}_maxlen_{max_len}_train_{train_json_path.split('/')[-1].split('.')[0]}_valid_{valid_json_path.split('/')[-1].split('.')[0]}_trainratio_{train_val_split}_time_{time_now}"
    wandb_logger = WandbLogger(log_model=log_model, project=project, name=run_name)

    kkr = KingKorre(rel2id_path=rel2id_path, max_token_len=max_len, mode=pooling_mode)
    tokenizer = kkr.tokenizer
    train_json_path = train_json_path
    valid_json_path = valid_json_path
    train_dataset = RelationshipExtractionDataset(
        train_json_path, tokenizer=tokenizer, max_len=512
    )
    if train_val_split is not None:
        train_size = int(train_val_split * len(train_dataset))
        valid_size = len(train_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, valid_size]
        )
    else:
        valid_dataset = RelationshipExtractionDataset(
            valid_json_path, tokenizer=tokenizer, max_len=512
        )
    print(f"train_dataset: {len(train_dataset)}")
    print(f"valid_dataset: {len(valid_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, run_name)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    trainer = l.Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        default_root_dir=save_path,
        devices=[gpu],
    )
    trainer.fit(
        model=kkr,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )


if __name__ == "__main__":
    Fire(main)
    # example usage: python3 train.py --project "KingKorre" --log_model "all" --rel2id_path "./gpt_relationships_only_person.json" --max_epochs 10 --train_json_path "sample_data/2대정종.json" --valid_json_path "sample_data/태조7월.json" --batch_size 16 --max_len 512
