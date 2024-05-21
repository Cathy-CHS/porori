from typing import List, Tuple
from entity import Entity, Linked_Entity
from itertools import permutations
from .korre import KorRE
from entity_extractor import Dotori
import pandas as pd
from tqdm import tqdm

# https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertModel
class Bono:
    def __init__(self):
        # korre load
        self.korre = KorRE()

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
        out = open("relations_output.txt", "w", encoding="utf-8")
        stringed_result = []
        heads = []
        tails = []
        relations = []
        for e in result_relations:
            heads.append(e[0])
            tails.append(e[1])
            relations.append(e[2])
            # out.write(str(e[0]), str(e[1]), e[2] + "\n")
            print(str(e[0]), str(e[1]), e[2])
            print(e)

        stringe_result_df = pd.DataFrame(
            {"head": heads, "tail": tails, "relation": relations}
        )
        stringe_result_df.to_csv("relations_output.csv", index=False)
        print("처리가 완료되었습니다. 결과는 {} 파일에 저장되었습니다.".format("relations_output.csv"))
        # out.close()

        # for debugging

        return result_relations
    
    def _process_chunk(self, chunk, chunk_start_idx, entities, result_relations):
        chunk_end_idx = chunk_start_idx + len(chunk)
        chunk_entities = [
            (entity, item)
            for entity in entities
            for item in entity.items
            if chunk_start_idx <= item["start"] < chunk_end_idx
        ]

        adjusted_entities = [
            {"entity": entity, "start": item["start"] - chunk_start_idx, "end": item["end"] - chunk_start_idx}
            for entity, item in chunk_entities
        ]

        entity_pairs = list(permutations(adjusted_entities, 2))

        for head, tail in entity_pairs:
            relation = self._relation_extract(chunk, head, tail)
            if relation:
                result_relations.extend(relation)


    def _relation_extract(self, document, head, tail):
        subj_range = [head["start"], head["end"]]
        obj_range = [tail["start"], tail["end"]]

        # Using markers to enhance the relation extraction
        marked_sentence = self.korre.entity_markers_added(
            document, subj_range, obj_range
        )
        print(marked_sentence)
        print("\n head:" + str(head["entity"]), str(head["start"]), str(head["start"]))
        print("\n tail:" + str(tail["entity"]), str(tail["start"]), str(tail["start"]))

        # Extract relations using the marked sentence
        relations = self.korre.infer(marked_sentence, entity_markers_included=True)

        # Convert the relation ID to relation name and map it with entities
        return [(head["entity"], tail["entity"], rel) for _, _, rel in relations]


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
