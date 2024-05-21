from src.relationship.data import RelationshipExtractionDataset
from src.relationship.korre import KingKorre
from torch.utils.data import DataLoader
import lightning as l

from lightning.pytorch.loggers import WandbLogger
from fire import Fire


def main(
    project: str = "KingKorre",
    log_model: str = "all",
    rel2id_path: str = "./gpt_relationships_only_person.json",
    max_epochs: int = 10,
    train_json_path: str = "sample_data/2대정종.json",
    valid_json_path: str = "sample_data/태조7월.json",
    batch_size: int = 16,
    max_len: int = 512,
):
    wandb_logger = WandbLogger(log_model=log_model, project=project)

    kkr = KingKorre(rel2id_path=rel2id_path, max_token_len=max_len)
    tokenizer = kkr.tokenizer
    train_json_path = train_json_path
    valid_json_path = valid_json_path
    train_dataset = RelationshipExtractionDataset(
        train_json_path, tokenizer=tokenizer, max_len=512
    )
    valid_dataset = RelationshipExtractionDataset(
        valid_json_path, tokenizer=tokenizer, max_len=512
    )
    print(f"train_dataset: {len(train_dataset)}")
    print(f"valid_dataset: {len(valid_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    trainer = l.Trainer(max_epochs=max_epochs, logger=wandb_logger)
    trainer.fit(
        model=kkr,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )


if __name__ == "__main__":
    Fire(main)
    # example usage: python train.py --project "KingKorre" --log_model "all" --rel2id_path "./gpt_relationships_only_person.json" --max_epochs 10 --train_json_path "sample_data/2대정종.json" --valid_json_path "sample_data/태조7월.json" --batch_size 16 --max_len 512
