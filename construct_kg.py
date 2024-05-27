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

import pandas as pd
from tqdm import tqdm
from time import time
from fire import Fire

load_dotenv()


def save_linked_entity_to_json(linked_entity_list, filename):
    # 객체를 딕셔너리로 변환

    with open(filename, "w", encoding="utf-8") as json_file:
        entity_dict_list = []
        for entity in linked_entity_list:
            entity_dict = {
                "name": entity.name,
                "entity_id": entity.entity_id,
                "items": entity.items,
            }
            entity_dict_list.append(entity_dict)

        json.dump(entity_dict_list, json_file, ensure_ascii=False, indent=4)
    print(f"Linked entities saved to {filename}")


def save_linked_entities_to_csv(
    linked_entities: List[Linked_Entity], filename: str
) -> None:
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

    df = pd.DataFrame({"Entity_name": entity_names, "Entity_id": entity_ids})
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Linked entities saved to {filename}")


import concurrent.futures


def process_entity(e, current_king="세종", hodu=None):
    if e.word == "임금" or e.word == "상":
        king_entity = Entity("PS", current_king, e.start, e.end)
        return hodu.get_id(king_entity)
    else:
        return hodu.get_id(e)


def main(
    current_king="선조",
    input_dir="input_texts/sunjo_1597",
    export_path="results/sunjo_1597",
):

    files = os.listdir(input_dir)
    print("Start Processing on the following files:")
    for file in files:
        print(file)
    if os.path.exists(export_path):
        ans = input("The export path already exists. Do you want to overwrite? (y/n)")
        if ans == "n":
            raise ValueError("Export path already exists. Exiting...")

    os.makedirs(export_path, exist_ok=True)
    start = time()
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

    # 3. Entity linking
    kb = EncyKoreaAPIKnowledgeBase()
    linked_entities = []

    hodu = Hodu(kb)
    get_id_results = []

    get_id_results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all tasks to the executor
        futures = {
            executor.submit(process_entity, e, current_king, hodu): e for e in entities
        }

        # Use tqdm to display the progress
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(entities),
            desc="Entity linking",
        ):
            try:
                result = future.result()
                get_id_results.append(result)
            except Exception as exc:
                print(f"Generated an exception: {exc}")
                get_id_results.append(None)

    entity_id_to_linked_entities = {}
    for e, result_entity in zip(entities, get_id_results):
        if result_entity is None:
            continue
        if result_entity.entity_id not in entity_id_to_linked_entities:
            new_linked_entity = Linked_Entity(
                result_entity.name, result_entity.entity_id
            )
            new_linked_entity.add_item(e.start, e.end)
            entity_id_to_linked_entities[result_entity.entity_id] = new_linked_entity

        else:
            existing_linked_entities = entity_id_to_linked_entities[
                result_entity.entity_id
            ]
            existing_linked_entities.add_item(e.start, e.end)

    linked_entities = list(entity_id_to_linked_entities.values())

    # f = open("linked_entities.txt", "w", encoding="utf-8")
    # for e in linked_entities:
    #     f.write(f"Entity: {e.name}, ID: {e.entity_id}")
    # f.close(
    save_linked_entity_to_json(
        linked_entities, os.path.join(export_path, "linked_entity.json")
    )
    save_linked_entities_to_csv(
        linked_entities, os.path.join(export_path, "linked_entities.csv")
    )
    # 4. Relation Extraction

    bono = Bono(
        kingkorre_model_path="pretrained_weight/kingkorre_all.ckpt", threshold=0.4
    )
    json_to_linked_entities = bono.load_linked_entities_from_json(
        os.path.join(export_path, "linked_entity.json")
    )
    result = bono.relation_extract(
        combined_text,
        json_to_linked_entities,
        512,
        os.path.join(export_path, "relationships.csv"),
    )
    print(f"Relationships saved to {os.path.join(export_path, 'relationships.csv')}")
    print(f"Processing time: {time()-start:.4f}s")


if __name__ == "__main__":
    Fire(main)
