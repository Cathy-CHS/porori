import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from relationship_extractor.relationship_extractor import Bono
from typing import List, Tuple
from entity import Entity
from itertools import permutations
from relationship_extractor.korre import KorRE
from entity_extractor import Dotori
import pandas as pd
from remove import NeoBuri


def main():
    # 1. load KB
    # 2. load siloc document
    # 3. generate knowledge graph
    # out = open("relations_output.txt", "w", encoding="utf-8")
    # 한자 제거
    input_file = 'src/input.txt'
    neoburi = NeoBuri(input_file)
    neoburi.process_text()

    # f = open("src/output.txt", "r", encoding="utf-8")
    # input_text = f.read()
    # dotori = Dotori()
    # entity_list = dotori.extract_entities(input_text)
    # grouped = dotori.group_chunk(entity_list)
    # filtered = dotori.filter_type(grouped)
    # entities = dotori.to_entity(filtered)
    # bono = Bono()

    # result = bono.extract(input_text, entities)

    # stringed_result = []
    # heads = []
    # tails = []
    # relations = []
    # for e in result:
    #     heads.append(e[0])
    #     tails.append(e[1])
    #     relations.append(e[2])
    #     # out.write(str(e[0]), str(e[1]), e[2] + "\n")
    #     print(str(e[0]), str(e[1]), e[2])
    #     print(e)

    # stringe_result_df = pd.DataFrame(
    #     {"head": heads, "tail": tails, "relation": relations}
    # )
    # stringe_result_df.to_csv("relations_output.csv", index=False)
    # f.close()
    # # out.close()


if __name__ == "__main__":
    main()
