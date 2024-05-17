import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from relationship_extractor.relationship_extractor import Bono
from typing import List, Tuple
from entity import Entity, Linked_Entity
from itertools import permutations
from relationship_extractor.korre import KorRE
from entity_extractor import Dotori
from remove import NeoBuri
from entity_linker.entity_linker import Hodu
from knowledgebase.knowledgebase import Knowledgebase, EncyKoreaAPIEntity
from dotenv import load_dotenv

load_dotenv()
def main():
    # 1. load KB
    # 2. load siloc document
    # 3. generate knowledge graph

    # 1. 한자 제거
    input_file = 'input.txt'
    neoburi = NeoBuri(input_file)
    processed_text = neoburi.process_text()

    # 2. Entity extraction
    dotori = Dotori()
    entities = dotori.extract_entities(processed_text, True)

    # 3. Entity linking
    kb = Knowledgebase()
    linked_entities = []

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
            new_entity.add_item(e['start'], e['end'])
            linked_entities.append(new_entity)
        else:
            existing_entity.add_item(e['start'], e['end'])

        # result는 EncyKoreaAPIEntity(name=item["headword"], entity_id=item["eid"], access_key=self.access_key) 로 반환
        # 만약에 hodu.get_id로 나온 result의 entity_id가 Linked_Entity 중에 존재하지 않으면 Linked_Entity 생성
        # e를 Linked_Entity의 items에 start, end index와 함께 넣어주기
        # Linked_Entity(name, id, item)
    print(linked_entities)
    # 4. Relation Extraction
    #bono = Bono()
    #result = bono.relation_extract(processed_text, linked_entities)



if __name__ == "__main__":
    main()
