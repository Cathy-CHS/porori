from typing import List, Tuple
from entity import Entity
from itertools import permutations
from korre import KorRE


# https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertModel
class Bono:
    def __init__(
        self,
    ):
        # korre load
        self.korre = KorRE()
        pass

    def extract(
        self, document: str, entities: List[Entity]
    ) -> List[Tuple[Entity, Entity, int]]:
        # permutation에 대해서 batch 처리
        result_relations = []

        sentences = document.split('. ')
        sentence_start_idx = 0

        for sentence in sentences:
            sentence_end_idx = sentence_start_idx + len(sentence)

            sentence_entities = [
                entity for entity in entities 
                if sentence_start_idx <= entity.start < sentence_end_idx
            ]

            adjusted_entities = [
                Entity(
                    entity=entity.entity, 
                    word=entity.word, 
                    start=entity.start - sentence_start_idx, 
                    end=entity.end - sentence_start_idx
                ) 
                for entity in sentence_entities
            ]

            entity_pairs = itertools.permutations(adjusted_entities, 2)

            for head, tail in entity_pairs:
                relation = self._extract(sentence, head, tail)
                if relation:
                    result_relations.extend(relation)

            sentence_start_idx = sentence_end_idx + 1 

        return result_relations

    def _extract(self, document, head, tail):
        subj_range = [head.start, head.end]
        obj_range = [tail.start, tail.end]

        # Using markers to enhance the relation extraction
        marked_sentence = self.korre.entity_markers_added(document, subj_range, obj_range)
        
        # Extract relations using the marked sentence
        relations = self.korre.infer(marked_sentence, entity_markers_included=True)
        
        # Convert the relation ID to relation name and map it with entities
        return [(head, tail, rel) for _, _, rel in relations]
