from src.relationship.korre import KingKorre
from src.relationship.data import RelationshipExtractionDataset
import os

if __name__ == "__main__":
    ckpt_path = "trained_models/best_model.ckpt"
    kkr = KingKorre.load_from_checkpoint(ckpt_path)
    json_path = "sample_data/2대정종.json"
    dataset = RelationshipExtractionDataset(
        json_path, tokenizer=kkr.tokenizer, max_len=512
    )
    sample = dataset[0]
    print(sample[0])

    print(kkr.predict(sample[0], get_labels=True))
    print(kkr.parse_id2class(sample[3]))

    # silocs_dir = "data/silocs"
    # silocs = os.listdir(silocs_dir)
    # silocs = [os.path.join(silocs_dir, siloc) for siloc in silocs]
    # print(silocs)

    # print(kkr.tokenizer.token_to_id("[CLS]"))
    # kkr.train_tokenizer(siloc_texts)
