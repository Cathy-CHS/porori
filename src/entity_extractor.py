import re
from transformers import AutoTokenizer, pipeline
from src.entity.entity import Entity
from typing import List
import torch

class Dotori:
    def __init__(self, max_tokens=512, device = None):
        # load Roberta Entity Recognition model
        # Use a pipeline as a high-level helper
        
        if torch.cuda.is_available():
            self.device = 0
        else:
            self.device = 'cpu'

        if device is not None:
            self.device = device
        self.pipe = pipeline(
            "token-classification", model="yongsun-yoon/klue-roberta-base-ner", device=self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "yongsun-yoon/klue-roberta-base-ner"
        )
        self.max_tokens = max_tokens
        self.processed_entities = []

    def extract_entities(self, text, filtered: bool = True):
        entity_list = self._extract_entities(text)
        grouped = self.group_chunk(entity_list)

        if filtered:  # filter 할 경우
            filtered = self.filter_type(grouped)
            entities = self.to_entity(filtered)
        else:  # filter 안 할 경우
            entities = self.to_entity(grouped)

        return entities

    def _extract_entities(self, text):
        # 문장으로 나누기
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = []
        current_length = 0

        # 문장들을 적절한 크기의 청크로 합치기
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            num_tokens = len(tokens)
            if current_length + num_tokens > self.max_tokens or len(current_chunk) >= 3:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = num_tokens
            else:
                current_chunk.append(sentence)
                current_length += num_tokens
        # 마지막 청크 추가
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        entities = []
        current_text_position = 0

        for chunk in chunks:
            chunk_entities = self.pipe(chunk)

            if "임금" in chunk:
                chunk_entities.append(
                    {
                        "entity": "PS",
                        "word": "임금",
                        "start": chunk.index("임금") + current_text_position,
                        "end": chunk.index("임금")
                        + current_text_position
                        + len("임금"),
                    }
                )

            entities.extend(
                {
                    "entity": entity["entity"],
                    "word": entity["word"],
                    "start": entity["start"] + current_text_position,
                    "end": entity["end"] + current_text_position,
                }
                for entity in chunk_entities
            )
            current_text_position += len(chunk) + 1

        return entities

    def to_entity(self, entities):
        return [
            Entity(entity["entity"], entity["word"], entity["start"], entity["end"])
            for entity in entities
        ]

    def group_chunk(self, entities):
        # I로 시작하는 같은 entity type의 chunk 묶기
        current_group = None

        for entity in entities:
            if entity["entity"].startswith("B-"):
                if current_group:
                    self.processed_entities.append(current_group)
                current_group = {
                    "entity": entity["entity"][2:],  # Remove 'B-' prefix
                    "word": entity["word"].replace("##", ""),  # Clean up word
                    "start": entity["start"],
                    "end": entity["end"],
                }
            elif entity["entity"].startswith("I-") and current_group:
                if (
                    entity["entity"][2:] == current_group["entity"]
                ):  # Check if the same entity type
                    current_group["word"] += entity["word"].replace("##", "")
                    current_group["end"] = entity["end"]
                else:  # Different entity type
                    if current_group:
                        self.processed_entities.append(current_group)
                    current_group = {
                        "entity": entity["entity"][2:],  # Remove 'I-' prefix
                        "word": entity["word"].replace("##", ""),  # Clean up word
                        "start": entity["start"],
                        "end": entity["end"],
                    }
                    # current_group = None  # Reset for safety

        if current_group:
            self.processed_entities.append(current_group)

        return self.processed_entities

    def filter_type(self, entities: List[dict], type: str = "PS"):
        filtered_list = []
        for entity in entities:
            if entity["entity"].startswith(type):
                filtered_list.append(entity)

        return filtered_list


if __name__ == "__main__":
    out = open("entities_output.txt", "w", encoding="utf-8")
    f = open("output.txt", "r", encoding="utf-8")

    text = f.read()
    dotori = Dotori()
    entities = dotori.extract_entities(text)
    result = dotori.group_chunk(entities)

    for e in entities:
        print(e)

    print("---------------------------------")

    for e in result:
        # print(e)
        out.write(str(e) + "\n")
    out.close()
