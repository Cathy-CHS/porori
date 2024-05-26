from typing import List, Tuple
import sys, os
import json

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from entity.entity import Entity, Linked_Entity
from itertools import permutations
from .korre import KingKorre, add_entity_markers
from entity_extractor import Dotori
import pandas as pd
from tqdm import tqdm


MAX_RELATIONS_ON_MEMORY = 10000
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

    def load_linked_entities_from_json(self, filename):
        with open(filename, 'r', encoding='utf-8') as json_file:
            entities_list = json.load(json_file)
        
        linked_entities = []
        for entity_dict in entities_list:
            entity = Linked_Entity(entity_dict['name'], entity_dict['entity_id'])
            entity.items = [tuple(item) for item in entity_dict['items']]
            linked_entities.append(entity)
        
        return linked_entities

    def relation_extract(
        self, document: str, entities: List[Linked_Entity], max_length: int, dump_file: str = "relations_output.csv"
    ) -> List[Tuple[Linked_Entity, Linked_Entity, int]]:
        # permutation에 대해서 batch 처리
        result_relations = []

        sentences = document.split(". ")
        current_chunk = ""
        current_chunk_start_idx = 0

        for sentence in tqdm(sentences):
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

            # Add dumping for memory issue.
            if len(result_relations) > MAX_RELATIONS_ON_MEMORY:
                self.save_triplets_to_json(result_relations, dump_file)
                result_relations = []
            
            

        # Process the last chunk
        if current_chunk:
            self._process_chunk(
                current_chunk, current_chunk_start_idx, entities, result_relations
            )
        
        self.save_triplets_to_json(result_relations, dump_file)
        result_relations = []
        
        return result_relations
    
    def save_triplets_to_json(self, relationships: List[Tuple[Linked_Entity, Linked_Entity, str]], filename:str) -> None:
        """
        Description:
            Save the relationships into csv file. The columns are head_entity_id(str), tail_entity_id(str), relation(str).
            After using this method, reinitialize the relationships list to [].
        
        Args:
            relationships: List[Tuple[Entity, Entity, int]].
            filename: str. The name of the file to save the relationships.
        """
        heads = []
        tails = []
        relations = []
        for r in relationships:
            heads.append(r[0].entity_id)
            tails.append(r[1].entity_id)
            relations.append(r[2])
        
        df = pd.DataFrame(
            {
                "head_entity_id": heads,
                "tail_entity_id": tails,
                "relation": relations
            }
        )
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=False, index=False)
            print(f"Relationship {filename} already exists. Appending to the file.")
        else:
            df.to_csv(filename, index=False)
            print(f"Relationship csv file: {filename} created successfully.")

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
        # print(f"Current Head and Tails are: {head_name}, {tail_name}")
        # subj_range = [(item[0], item[1]) for item in head["entity"].items]
        # obj_range = [(item[0], item[1]) for item in tail["entity"].items]

        # Using markers to enhance the relation extraction
        marked_sentence = add_entity_markers(document, head_idx, tail_idx)
        # print(marked_sentence)
        # print("\n head:" + str(head["entity"]), str(head["start"]), str(head["start"]))
        # print("\n tail:" + str(tail["entity"]), str(tail["start"]), str(tail["start"]))

        # Extract relations using the marked sentence

        labels, classes = self.korre.predict(
            marked_sentence, get_labels=True, conf_threshold=self.threshold
        )
        # logits = self.korre.predict(marked_sentence, get_labels=False)
        # print("Logits: ", logits)

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
