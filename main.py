import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from relationship.relationship_extractor import Bono
from typing import List, Tuple
from entity import Entity, Linked_Entity
from itertools import permutations
from relationship.korre import KorRE
from entity_extractor import Dotori
from remove import NeoBuri
from entity_linker.entity_linker import Hodu
from knowledgebase.knowledgebase import EncyKoreaAPIKnowledgeBase
from dotenv import load_dotenv

load_dotenv()
def main():
    # 1. load KB
    # 2. load siloc document
    # 3. generate knowledge graph
    # out = open("relations_output.txt", "w", encoding="utf-8")

    # 한자 제거
    input_file = 'src/input.txt'
    neoburi = NeoBuri(input_file)
    neoburi.process_text()

    # 1. 한자 제거
    input_dir = 'input_texts/연산 1년 1월' #인풋 디렉토리
    files = os.listdir(input_dir)

    combined_text = ' '.join(files)
    
    # # input_file = 'input.txt'
    # # neoburi = NeoBuri(input_file)
    # # processed_text = neoburi.process_text()
    print(combined_text)

    # 2. Entity extraction
#     dotori = Dotori()
#     entities = dotori.extract_entities(combined_text, True)
#     for e in entities:
#         print(str(e))

#     # 3. Entity linking
#     kb = EncyKoreaAPIKnowledgeBase()
#     linked_entities = []
#     existing_entity = None

#     hodu = Hodu(kb)
#     for e in entities:
#         result = hodu.get_id(e)

#         if result == None:
#             continue
        
#         for linked_entity in linked_entities:
#             if linked_entity.entity_id == result.entity_id:
#                 existing_entity = linked_entity
#                 break
        
#         if not existing_entity:
#             new_entity = Linked_Entity(result.name, result.entity_id)
#             new_entity.add_item(e.start, e.end)
#             linked_entities.append(new_entity)
#         else:
#             existing_entity.add_item(e.start, e.end)

#     # f = open("linked_entities.txt", "w", encoding="utf-8")
#     # for e in linked_entities:
#     #     f.write(f"Entity: {e.name}, ID: {e.entity_id}")
#     # f.close()

#     # 4. Relation Extraction
#     bono = Bono()
#     result = bono.relation_extract(combined_text, linked_entities, 1024)
    



if __name__ == "__main__":
    main()
