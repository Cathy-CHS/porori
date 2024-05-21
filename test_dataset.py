from src.relationship.data import RelationshipExtractionDataset
from src.relationship.korre import KingKorre
from torch.utils.data import Dataset, DataLoader
import lightning as l
import torch

if __name__ == "__main__":
    kkr = KingKorre(rel2id_path="./gpt_relationships_only_person.json")
    tokenizer = kkr.tokenizer
    dataset = RelationshipExtractionDataset(
        "src/rel_ext_data.json", tokenizer=tokenizer, max_len=512
    )
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    trainer = l.Trainer(max_epochs=3, accelerator="mps", devices=1)
    trainer.fit(model=kkr, train_dataloaders=dataloader)
