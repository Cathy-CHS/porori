import re
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from .korre import add_entity_markers


class RelationshipExtractionDataset(Dataset):
    """
    Multi-label classification dataset for Korean Relation Extraction.
    outputs form of (input_txt, input_ids, attention_mask, label)
    input_txt (str): input text
    input_ids (torch.Tensor): input token ids
    attention_mask (torch.Tensor): attention mask
    label (torch.Tensor): multi-label tensor. 1 if the relation exists, 0 otherwise.
    since we have 65 classes, the label tensor has shape of (65,)

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, data_path, tokenizer, max_len=512, num_classes=65):
        with open(data_path, "r", encoding="utf-8-sig") as f:
            self.data = json.load(f)["data"]
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.special_tokens_dict = {
            "additional_special_tokens": ["[CLS]", "[E1]", "[/E1]", "[E2]", "[/E2]"]
        }
        self.tokenizer.add_special_tokens(self.special_tokens_dict)

        self.data_list = []  # ((head, tail), text, label) list
        rel_dict = {}  # {(head, tail): [relationships]} dictionary
        for doc in self.data:
            for rel in doc["relationships"]:
                if (rel[0], rel[1]) in rel_dict:
                    rel_dict[(rel[0], rel[1])].append(rel[2])
                else:
                    rel_dict[(rel[0], rel[1])] = [rel[2]]
            for key, values in rel_dict.items():
                label = torch.zeros(num_classes)
                for value in values:
                    label[self.tokenizer.convert_tokens_to_ids(value)] = 1
                self.data_list.append((key, doc["input_text"], label))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        rel, text, label = self.data_list[index]
        head, tail = rel

        head_indices = []
        tail_indices = []

        head_entity = re.escape(head)
        tail_entity = re.escape(tail)

        head_matches = [(m.start(), m.end()) for m in re.finditer(head_entity, text)]
        tail_matches = [(m.start(), m.end()) for m in re.finditer(tail_entity, text)]

        head_indices.extend(head_matches)
        tail_indices.extend(tail_matches)
        text_with_markers = add_entity_markers(
            text, head_indices, tail_indices
        )  # Add entity markers to text

        encoding = self.tokenizer.encode_plus(
            text_with_markers,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return text_with_markers, input_ids, attention_mask, label
