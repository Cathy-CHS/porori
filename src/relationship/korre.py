import os
import sys
import torch
import numpy as np
import pandas as pd
import easydict
import argparse
import json
import requests
import wget


import warnings

warnings.filterwarnings("ignore")
from model import KREModel

# from pororo import Pororo
import os
import torch
import torch.nn as nn
import numpy as np
import json
import wget
import easydict
import logging
import lightning as L
from transformers import BertTokenizer, BertModel


def add_entity_markers(self, text, heads: torch.Tensor, tails: torch.Tensor):
    """
    Add [E1], [/E1], [E2], [/E2] tokens to the sentence based on the head and tail indices.
    There can be multiple occurrences of head and tail entities in the sentence. All
    the head and tail entities will be marked with [E1], [/E1], [E2], [/E2].

    Args:
        text (str): The input text.
        heads (torch.Tensor): Tensor of shape (num_heads, 2) containing start and end indices of head entities.
        tails (torch.Tensor): Tensor of shape (num_tails, 2) containing start and end indices of tail entities.
    """
    # Create a list of indices and markers
    indices = []

    for start, end in heads.tolist():
        indices.append((start, "[E1]"))
        indices.append((end + 1, "[/E1]"))

    for start, end in tails.tolist():
        indices.append((start, "[E2]"))
        indices.append((end + 1, "[/E2]"))

    # Sort the indices in reverse order to avoid messing up the positions after insertion
    indices.sort(reverse=True, key=lambda x: x[0])

    # Insert markers into the text
    for index, marker in indices:
        text = text[:index] + marker + text[index:]

    return text


class KorRE:
    def __init__(self):
        self.args = easydict.EasyDict(
            {
                "bert_model": "datawhales/korean-relation-extraction",
                "mode": "ALLCC",
                "n_class": 97,
                "max_token_len": 512,
                "max_acc_threshold": 0.6,
            }
        )
        # self.ner_module = Pororo(task='ner', lang='ko')

        logging.set_verbosity_error()

        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_model)

        # # entity markers tokens
        # special_tokens_dict = {'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]']}
        # num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)   # num_added_toks: 4

        self.trained_model = self.__get_model()

        # relation id to label
        r = requests.get(
            "https://raw.githubusercontent.com/datawhales/Korean_RE/main/data/relation/relid2label.json"
        )
        self.relid2label = json.loads(r.text)

        # relation list
        self.relation_list = list(self.relid2label.keys())

        # device
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.trained_model = self.trained_model.to(self.device)

    def __get_model(self):
        """사전학습된 한국어 관계 추출 모델을 로드하는 함수."""
        if not os.path.exists("./pretrained_weight"):
            os.mkdir("./pretrained_weight")

        pretrained_weight = "./pretrained_weight/pytorch_model.bin"

        if not os.path.exists(pretrained_weight):
            url = "https://huggingface.co/datawhales/korean-relation-extraction/resolve/main/pytorch_model.bin"
            wget.download(url, out=pretrained_weight)

        trained_model = KREModel(self.args)

        trained_model.load_state_dict(torch.load(pretrained_weight), strict=False)
        trained_model.eval()

        return trained_model

    def __idx2relid(self, idx_list):
        """onehot label에서 1인 위치 인덱스 리스트를 relation id 리스트로 변환하는 함수.

        Example:
            relation_list = ['P17', 'P131', 'P530', ...] 일 때,
            __idx2relid([0, 2]) => ['P17', 'P530'] 을 반환.
        """
        label_out = []

        for idx in idx_list:
            label = self.relation_list[idx]
            label_out.append(label)

        return label_out

    def entity_markers_added(
        self, sentence: str, subj_range: list, obj_range: list
    ) -> str:
        """문장과 관계를 구하고자 하는 두 개체의 인덱스 범위가 주어졌을 때 entity marker token을 추가하여 반환하는 함수.

        Example:
            sentence = '모토로라 레이저 M는 모토로라 모빌리티에서 제조/판매하는 안드로이드 스마트폰이다.'
            subj_range = [0, 10]   # sentence[subj_range[0]: subj_range[1]] => '모토로라 레이저 M'
            obj_range = [12, 21]   # sentence[obj_range[0]: obj_range[1]] => '모토로라 모빌리티'

        Return:
            '[E1] 모토로라 레이저 M [/E1] 는  [E2] 모토로라 모빌리티 [/E2] 에서 제조/판매하는 안드로이드 스마트폰이다.'
        """
        result_sent = ""

        for i, char in enumerate(sentence):
            if i == subj_range[0]:
                result_sent += " [E1] "
            elif i == subj_range[1]:
                result_sent += " [/E1] "
            if i == obj_range[0]:
                result_sent += " [E2] "
            elif i == obj_range[1]:
                result_sent += " [/E2] "
            result_sent += sentence[i]
        if subj_range[1] == len(sentence):
            result_sent += " [/E1]"
        elif obj_range[1] == len(sentence):
            result_sent += " [/E2]"

        return result_sent.strip()

    def infer(
        self,
        sentence: str,
        subj_range=None,
        obj_range=None,
        entity_markers_included=False,
    ):
        """입력받은 문장에 대해 관계 추출 태스크를 수행하는 함수."""
        # entity marker token이 포함된 경우
        if entity_markers_included:
            # subj, obj name 구하기
            tmp_input_ids = self.tokenizer(sentence)["input_ids"]

            if (
                tmp_input_ids.count(20000) != 1
                or tmp_input_ids.count(20001) != 1
                or tmp_input_ids.count(20002) != 1
                or tmp_input_ids.count(20003) != 1
            ):
                raise Exception(
                    "Incorrect number of entity marker tokens('[E1]', '[/E1]', '[E2]', '[/E2]')."
                )

            subj_start_id, subj_end_id = tmp_input_ids.index(
                20000
            ), tmp_input_ids.index(20001)
            obj_start_id, obj_end_id = tmp_input_ids.index(20002), tmp_input_ids.index(
                20003
            )

            subj_name = self.tokenizer.decode(
                tmp_input_ids[subj_start_id + 1 : subj_end_id]
            )
            obj_name = self.tokenizer.decode(
                tmp_input_ids[obj_start_id + 1 : obj_end_id]
            )

            encoding = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=self.args.max_token_len,
                return_token_type_ids=False,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            mask = encoding["attention_mask"].to(self.device)

            _, prediction = self.trained_model(input_ids, mask)

            predictions = [prediction.flatten()]
            predictions = torch.stack(predictions).detach().cpu()

            y_pred = predictions.numpy()
            upper, lower = 1, 0
            y_pred = np.where(y_pred > self.args.max_acc_threshold, upper, lower)

            preds_list = []

            for i in range(len(y_pred)):
                class_pred = self.__idx2relid(np.where(y_pred[i] == 1)[0])
                preds_list.append(class_pred)

            preds_list = preds_list[0]

            pred_rel_list = [self.relid2label[pred] for pred in preds_list]

            return [(subj_name, obj_name, pred_rel) for pred_rel in pred_rel_list]

        # entity_markers_included=False인 경우
        else:
            # entity marker가 문장에 포함된 경우
            tmp_input_ids = self.tokenizer(sentence)["input_ids"]

            if (
                tmp_input_ids.count(20000) >= 1
                or tmp_input_ids.count(20001) >= 1
                or tmp_input_ids.count(20002) >= 1
                or tmp_input_ids.count(20003) >= 1
            ):
                raise Exception(
                    "Entity marker tokens already exist in the input sentence. Try 'entity_markers_included=True'."
                )

            # subj range와 obj range가 주어진 경우
            if subj_range is not None and obj_range is not None:
                # add entity markers
                converted_sent = self.entity_markers_added(
                    sentence, subj_range, obj_range
                )

                encoding = self.tokenizer.encode_plus(
                    converted_sent,
                    add_special_tokens=True,
                    max_length=self.args.max_token_len,
                    return_token_type_ids=False,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                input_ids = encoding["input_ids"].to(self.device)
                mask = encoding["attention_mask"].to(self.device)

                _, prediction = self.trained_model(input_ids, mask)

                predictions = [prediction.flatten()]
                predictions = torch.stack(predictions).detach().cpu()

                y_pred = predictions.numpy()
                upper, lower = 1, 0
                y_pred = np.where(y_pred > self.args.max_acc_threshold, upper, lower)

                preds_list = []

                for i in range(len(y_pred)):
                    class_pred = self.__idx2relid(np.where(y_pred[i] == 1)[0])
                    preds_list.append(class_pred)

                preds_list = preds_list[0]

                pred_rel_list = [self.relid2label[pred] for pred in preds_list]

                return [
                    (
                        sentence[subj_range[0] : subj_range[1]],
                        sentence[obj_range[0] : obj_range[1]],
                        pred_rel,
                    )
                    for pred_rel in pred_rel_list
                ]

            # 문장만 주어진 경우: 모든 경우에 대해 inference 수행
            else:
                input_list = self.get_all_inputs(sentence)

                converted_sent_list = [
                    self.entity_markers_added(*input_list[i])
                    for i in range(len(input_list))
                ]

                encoding_list = []

                for i, converted_sent in enumerate(converted_sent_list):
                    tmp_encoding = self.tokenizer.encode_plus(
                        converted_sent,
                        add_special_tokens=True,
                        max_length=self.args.max_token_len,
                        return_token_type_ids=False,
                        padding="max_length",
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors="pt",
                    )
                    encoding_list.append(tmp_encoding)

                predictions = []

                for i, item in enumerate(encoding_list):
                    _, prediction = self.trained_model(
                        item["input_ids"].to(self.device),
                        item["attention_mask"].to(self.device),
                    )

                    predictions.append(prediction.flatten())

                if predictions:
                    predictions = torch.stack(predictions).detach().cpu()

                    y_pred = predictions.numpy()
                    upper, lower = 1, 0
                    y_pred = np.where(
                        y_pred > self.args.max_acc_threshold, upper, lower
                    )

                    preds_list = []
                    for i in range(len(y_pred)):
                        class_pred = self.__idx2relid(np.where(y_pred[i] == 1)[0])
                        preds_list.append(class_pred)

                    result_list = []
                    for i, input_i in enumerate(input_list):
                        tmp_subj_range, tmp_obj_range = input_i[1], input_i[2]
                        result_list.append(
                            (
                                sentence[tmp_subj_range[0] : tmp_subj_range[1]],
                                sentence[tmp_obj_range[0] : tmp_obj_range[1]],
                                preds_list[i],
                            )
                        )

                    final_list = []
                    for tmp_subj, tmp_obj, tmp_list in result_list:
                        for i in range(len(tmp_list)):
                            final_list.append((tmp_subj, tmp_obj, tmp_list[i]))

                    return [
                        (item[0], item[1], self.relid2label[item[2]])
                        for item in final_list
                    ]

                else:
                    return []


class KingKorre(L.LightningModule):
    def __init__(
        self,
        model_path: str = None,
        rel2id_path: str = "gpt_relationships_only_person.json",
        retrain: bool = True,
    ):
        super().__init__()

        self.mode = "max"
        self.args = easydict.EasyDict(
            {
                "bert_model": "datawhales/korean-relation-extraction",
                "mode": "ALLCC",
                "n_class": 65,
                "max_token_len": 512,
                "max_acc_threshold": 0.6,
            }
        )

        self.tokenizer = BertTokenizer.from_pretrained(
            "datawhales/korean-relation-extraction"
        )

        # Add entity markers tokens
        if retrain:
            special_tokens_dict = {
                "additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]
            }
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        self.trained_model = self.__get_korre_model()

        # relation id to label
        with open(rel2id_path, "r", encoding="utf-8") as f:
            self.relid2label = json.load(f)

        # relation list
        self.relation_list = list(self.relid2label.keys())

        # device
        # self.device = torch.device(
        #     "cuda"
        #     if torch.cuda.is_available()
        #     else "mps" if torch.backends.mps.is_available() else "cpu"
        # )

        # self.trained_model = self.trained_model.to(self.device)

    def __get_korre_model(self):
        """Load the pre-trained Korean relation extraction model."""
        if not os.path.exists("./pretrained_weight"):
            os.mkdir("./pretrained_weight")

        pretrained_weight = "./pretrained_weight/pytorch_model.bin"

        if not os.path.exists(pretrained_weight):
            url = "https://huggingface.co/datawhales/korean-relation-extraction/resolve/main/pytorch_model.bin"
            wget.download(url, out=pretrained_weight)

        trained_model = BertModel.from_pretrained(
            self.args.bert_model, return_dict=True
        )
        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_model)
        trained_model.load_state_dict(torch.load(pretrained_weight), strict=False)
        self.pretrained_config = trained_model.config

        # Define a separate classifier layer
        self.classifier = nn.Linear(trained_model.config.hidden_size, self.args.n_class)

        trained_model.eval()

        return trained_model

    def add_entity_markers(self, text, heads: torch.Tensor, tails: torch.Tensor):
        """
        Add [E1], [/E1], [E2], [/E2] tokens to the sentence based on the head and tail indices.
        There can be multiple occurrences of head and tail entities in the sentence. All
        the head and tail entities will be marked with [E1], [/E1], [E2], [/E2].

        Args:
            text (str): The input text.
            heads (torch.Tensor): Tensor of shape (num_heads, 2) containing start and end indices of head entities.
            tails (torch.Tensor): Tensor of shape (num_tails, 2) containing start and end indices of tail entities.
        """
        # Create a list of indices and markers
        indices = []

        for start, end in heads.tolist():
            indices.append((start, "[E1]"))
            indices.append((end + 1, "[/E1]"))

        for start, end in tails.tolist():
            indices.append((start, "[E2]"))
            indices.append((end + 1, "[/E2]"))

        # Sort the indices in reverse order to avoid messing up the positions after insertion
        indices.sort(reverse=True, key=lambda x: x[0])

        # Insert markers into the text
        for index, marker in indices:
            text = text[:index] + marker + text[index:]

        return text

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.trained_model(input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_outputs.last_hidden_state

        if self.mode == "max":
            # Max pooling
            pooled_output, _ = torch.max(last_hidden_state, dim=1)
        elif self.mode == "mean":
            # Mean pooling
            pooled_output = torch.mean(last_hidden_state, dim=1)
        elif self.mode == "cls":
            # CLS token pooling
            pooled_output = last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        logits = self.classifier(pooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self.forward(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self.forward(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        return optimizer


if __name__ == "__main__":
    king = KingKorre()