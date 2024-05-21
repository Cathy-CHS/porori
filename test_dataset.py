from src.relationship.data import RelationshipExtractionDataset
from src.relationship.korre import KingKorre

if __name__ == "__main__":
    kkr = KingKorre(rel2id_path="./gpt_relationships_only_person.json")
    tokenizer = kkr.tokenizer
    dataset = RelationshipExtractionDataset(
        "src/rel_ext_data.json", tokenizer=tokenizer, max_len=512
    )
    print(dataset[0])
    print(dataset[1])
