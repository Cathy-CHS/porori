import os, sys
import json

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from relationship.relationship_extractor import Bono
from typing import List, Tuple
from entity.entity import Entity, Linked_Entity
from itertools import permutations
from entity_extractor import Dotori
from remove import NeoBuri
from entity_linker.entity_linker import Hodu
from knowledgebase.knowledgebase import EncyKoreaAPIKnowledgeBase
from dotenv import load_dotenv
from visual import visual


load_dotenv()


def main():

    # input 받아서 합치기
    input_dir = "input_texts/연산 1년 1월"  # 인풋 디렉토리
    files = os.listdir(input_dir)

    texts = []
    for file in files:
        file_path = os.path.join(input_dir, file)
        f = open(file_path, "r", encoding="utf-8")
        text = f.read()
        texts.append(text)

    combined_text = " ".join(texts)

    # 2. Entity extraction
    dotori = Dotori()
    entities = dotori.extract_entities(combined_text, True)
    # for e in entities:
    #     print(str(e))

    # 3. Entity linking
    kb = EncyKoreaAPIKnowledgeBase()
    linked_entities = []
    existing_entity = None

    hodu = Hodu(kb)
    for e in entities:
        result = hodu.get_id(e)

        if result == None:
            continue

        for linked_entity in linked_entities:
            if linked_entity.entity_id == result.entity_id:
                existing_entity = linked_entity
                break

        if not existing_entity:
            new_entity = Linked_Entity(result.name, result.entity_id)
            new_entity.add_item(e.start, e.end)
            linked_entities.append(new_entity)
        else:
            existing_entity.add_item(e.start, e.end)

    # f = open("linked_entities.txt", "w", encoding="utf-8")
    # for e in linked_entities:
    #     f.write(f"Entity: {e.name}, ID: {e.entity_id}")
    # f.close()
    hodu.save_linked_entity_to_json(linked_entity, 'linked_entity.json')

    # 4. Relation Extraction

    bono = Bono(kingkorre_model_path="pretrained_weight/kingkorre_all.ckpt", threshold=0.1)
    json_to_linked_entities = bono.load_linked_entities_from_json('linked_entity.json')
    result = bono.relation_extract(combined_text, json_to_linked_entities, 512)

    # 5. Construct Knowledge graph
    visual(result)


if __name__ == "__main__":
    main()
