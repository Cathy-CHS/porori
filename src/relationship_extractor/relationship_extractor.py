from typing import List, Tuple
from entity import Entity


# https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertModel
class Bono:
    def __init__(
        self,
    ):
        # korre load
        pass

    def extract(
        self, document: str, entities: List[Entity]
    ) -> List[Tuple[Entity, Entity, int]]:
        # permutation에 대해서 batch 처리
        pass

    def _extract(self, document, head, tail):
        pass
