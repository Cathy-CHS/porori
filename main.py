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
import pandas as pd
from tqdm import tqdm

load_dotenv()

def save_linked_entity_to_json(linked_entity_list, filename):
        # 객체를 딕셔너리로 변환
        
    with open(filename, 'w', encoding='utf-8') as json_file:
        entity_dict_list = []
        for entity in linked_entity_list:
            entity_dict = {
                'name': entity.name,
                'entity_id': entity.entity_id,
                'items': entity.items
            }
            entity_dict_list.append(entity_dict)

        json.dump(entity_dict_list, json_file, ensure_ascii=False, indent=4)
    print(f"Linked entities saved to {filename}")

def save_linked_entities_to_csv(linked_entities: List[Linked_Entity], filename:str) -> None:
    """
    Save linked entities to a CSV file. The CSV file will have the following columns:
    - Entity_name (str)
    - Entity_id (str)



    Args:
        linked_entities (_type_): _description_
        filename (_type_): _description_
    """
    entity_names = []
    entity_ids = []
    for entity in linked_entities:
        entity_names.append(entity.name)
        entity_ids.append(entity.entity_id)
    
    df = pd.DataFrame({
        'Entity_name': entity_names,
        'Entity_id': entity_ids
    })
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Linked entities saved to {filename}")

def main():

    # input 받아서 합치기
    current_king = "세종"
    input_dir = "input_texts/test"  # 인풋 디렉토리
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
    for e in tqdm(entities, desc="Entity linking"):
        if e.word =="임금":
            king_entity = Entity("PS", current_king, e.start, e.end)
            result = hodu.get_id(king_entity)
        else:
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
    # f.close(
    save_linked_entity_to_json(linked_entities, 'linked_entity.json')
    save_linked_entities_to_csv(linked_entities, 'linked_entities.csv')
    # 4. Relation Extraction

    bono = Bono(kingkorre_model_path="pretrained_weight/kingkorre_all.ckpt", threshold=0.85)
    json_to_linked_entities = bono.load_linked_entities_from_json('linked_entity.json')
    result = bono.relation_extract(combined_text, json_to_linked_entities, 512, "세종실록.csv")
    for e in result:
        print(e)
    # 5. Construct Knowledge graph
    # visual(result)


if __name__ == "__main__":
    main()
