from src.relationship.data import RelationshipExtractionDataset
from src.relationship.korre import KingKorre
from torch.utils.data import DataLoader
import lightning as l


if __name__ == "__main__":
    kkr = KingKorre(rel2id_path="./gpt_relationships_only_person.json")
    tokenizer = kkr.tokenizer
    train_json_path = "src/rel_ext_data.json"
    valid_json_path = "src/rel_ext_data.json"
    train_dataset = RelationshipExtractionDataset(
        train_json_path, tokenizer=tokenizer, max_len=512
    )
    valid_dataset = RelationshipExtractionDataset(
        valid_json_path, tokenizer=tokenizer, max_len=512
    )
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
    trainer = l.Trainer(max_epochs=3, accelerator="mps", devices=1)
    trainer.fit(
        model=kkr, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader
    )
