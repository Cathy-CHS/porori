from typing import List, Tuple
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from entity.entity import Entity, Linked_Entity
from itertools import permutations
from .korre import KingKorre, add_entity_markers
from entity_extractor import Dotori
import pandas as pd
from tqdm import tqdm


# https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertModel
class Bono:
    def __init__(self, kingkorre_model_path: str = None, threshold=0.8):
        # korre load
        if kingkorre_model_path:
            self.korre = KingKorre.load_from_checkpoint(kingkorre_model_path)
            print("KingKorre model loaded successfully!")
        else:
            self.korre = KingKorre()
        self.threshold = threshold

    def relation_extract(
        self, document: str, entities: List[Entity], max_length: int
    ) -> List[Tuple[Entity, Entity, int]]:
        # permutation에 대해서 batch 처리
        result_relations = []

        sentences = document.split(". ")
        current_chunk = ""
        current_chunk_start_idx = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
            else:
                # Process the current chunk
                self._process_chunk(
                    current_chunk, current_chunk_start_idx, entities, result_relations
                )

                # Start new chunk
                current_chunk_start_idx += len(current_chunk) + 2
                current_chunk = sentence

        # Process the last chunk
        if current_chunk:
            self._process_chunk(
                current_chunk, current_chunk_start_idx, entities, result_relations
            )

        # for debugging
        # out = open("relations_output.txt", "w", encoding="utf-8")
        stringed_result = []
        heads = []
        tails = []
        relations = []
        for e in result_relations:
            heads.append(e[0])
            tails.append(e[1])
            relations.append(e[2])
            # out.write(str(e[0]), str(e[1]), e[2] + "\n")
            # print(str(e[0]), str(e[1]), e[2])
            # print(e)

        stringe_result_df = pd.DataFrame(
            {"head": heads, "tail": tails, "relation": relations}
        )
        stringe_result_df.to_csv("relations_output.csv", index=False)
        print(
            "처리가 완료되었습니다. 결과는 {} 파일에 저장되었습니다.".format(
                "relations_output.csv"
            )
        )
        # out.close()

        # for debugging

        return result_relations

    def _process_chunk(
        self,
        chunk: str,
        chunk_start_idx: int,
        entities: List[Linked_Entity],
        result_relations: List,
    ):
        chunk_end_idx = chunk_start_idx + len(chunk)
        # chunk_entities = [
        #     (entity, item)
        #     for entity in entities
        #     for item in entity.items
        #     if (item[1] < chunk_end_idx) and (item[0] >= chunk_start_idx)
        # ]

        entities_in_chunk = {}
        for entity in entities:
            entity_in_chunk = [
                (item[0] - chunk_start_idx, item[1] - chunk_start_idx)
                for item in entity.items
                if ((item[1] < chunk_end_idx) and (item[0] >= chunk_start_idx))
            ]
            if len(entity_in_chunk) > 0:
                entities_in_chunk[entity.name] = entity_in_chunk

        # adjusted_entities = [
        #     {
        #         "entity": entity,
        #         "start": item[0] - chunk_start_idx,
        #         "end": item[1] - chunk_start_idx,
        #     }
        #     for entity, item in chunk_entities
        # ]

        entity_pairs = list(permutations(list(entities_in_chunk.keys()), 2))

        for head, tail in entity_pairs:
            relation = self._relation_extract(
                chunk, head, tail, entities_in_chunk[head], entities_in_chunk[tail]
            )
            if relation:
                result_relations.extend(relation)

    def _relation_extract(
        self,
        document: str,
        head_name: str,
        tail_name: str,
        head_idx: List[Tuple[int, int]],
        tail_idx: List[Tuple[int, int]],
    ):
        print(f"Current Head and Tails are: {head_name}, {tail_name}")
        # subj_range = [(item[0], item[1]) for item in head["entity"].items]
        # obj_range = [(item[0], item[1]) for item in tail["entity"].items]

        # Using markers to enhance the relation extraction
        marked_sentence = add_entity_markers(document, head_idx, tail_idx)
        print(marked_sentence)
        # print("\n head:" + str(head["entity"]), str(head["start"]), str(head["start"]))
        # print("\n tail:" + str(tail["entity"]), str(tail["start"]), str(tail["start"]))

        # Extract relations using the marked sentence

        labels, classes = self.korre.predict(
            marked_sentence, get_labels=True, conf_threshold=self.threshold
        )
        logits = self.korre.predict(marked_sentence, get_labels=False)
        print("Logits: ", logits)

        # Convert the relation ID to relation name and map it with entities
        return [(head_name, tail_name, rel) for rel in classes]


if __name__ == "__main__":
    f = open("output.txt", "r", encoding="utf-8")
    input_text = f.read()
    dotori = Dotori()
    entity_list = dotori.extract_entities(input_text)
    result = dotori.group_chunk(entity_list)
    entities = dotori.to_entity(result)
    bono = Bono()

    result = bono.relation_extract(input_text, entities)
    # for e in result:
    #     print(e)
    f.close()
