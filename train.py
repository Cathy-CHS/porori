from src.relationship.data import RelationshipExtractionDataset
from src.relationship.korre import KingKorre
from torch.utils.data import DataLoader
import lightning as l

from lightning.pytorch.loggers import WandbLogger


if __name__ == "__main__":
    wandb_logger = WandbLogger(log_model="all")

    kkr = KingKorre(rel2id_path="./gpt_relationships_only_person.json")
    tokenizer = kkr.tokenizer
    train_json_path = "sample_data/2대정종.json"
    valid_json_path = "sample_data/태조7월.json"
    train_dataset = RelationshipExtractionDataset(
        train_json_path, tokenizer=tokenizer, max_len=512
    )
    valid_dataset = RelationshipExtractionDataset(
        valid_json_path, tokenizer=tokenizer, max_len=512
    )
    print(f"train_dataset: {len(train_dataset)}")
    print(f"valid_dataset: {len(valid_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    trainer = l.Trainer(max_epochs=1, logger=wandb_logger)
    trainer.fit(
        model=kkr,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
