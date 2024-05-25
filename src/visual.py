import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx
import os

# import csv
# from entity.entity import Entity


def visual(data, output_path="graph.png"):
    # font_path = os.path.join("fonts", "NANUMGOTHIC.TTF")  # 한국어 폰트 파일 경로
    # fontprop = fm.FontProperties(fname=font_path)
    # plt.rc("font", family=fontprop.get_name())

    G = nx.DiGraph()

    for head, tail, relation in data:
        G.add_node(head)
        G.add_node(tail)
        G.add_edge(head, tail, label=relation)

    # 엣지 라벨을 포함한 그래프 그리기
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=3000,
        node_color="lightblue",
        font_size=10,
        font_weight="bold",
        arrows=False,
        edge_color="gray",
        width=2,
        # font_family=fontprop.get_name(),
    )

    # 엣지 라벨 추가
    edge_labels = {(head, tail): relation for head, tail, relation in data}
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_color="red",
        # font_family=fontprop.get_name(),
    )

    plt.title(
        "Knowledge Graph",  
        #   fontproperties=fontprop
    )
    plt.savefig(
	'graph.png', dpi=300, facecolor='white', edgecolor='black',
    orientation='portrait', format='png', transparent=False,
    bbox_inches='tight', pad_inches=0.1)
    # plt.savefig(output_path, format="png", dpi=300)
    plt.show()
    plt.close()


# # 데이터 파싱 및 Entity 객체 생성
# def parse_entity(entity_str: str) -> Entity:
#     parts = entity_str.split(", ")
#     entity = parts[0].split(": ")[1]
#     word = parts[1].split(": ")[1]
#     start = int(parts[2].split(": ")[1])
#     end = int(parts[3].split(": ")[1])
#     return Entity(entity, word, start, end)


# data = []
# with open(csv_file, newline="", encoding="utf-8") as csvfile:
#     reader = csv.reader(csvfile)
#     next(reader)  # 첫 번째 행 (헤더) 건너뛰기
#     for row in reader:
#         head_str, tail_str, relation = row
#         head = parse_entity(head_str)
#         tail = parse_entity(tail_str)
#         data.append((head, tail, relation))

# G = nx.DiGraph()

# for head, tail, relation in data:
#     G.add_node(head.word)
#     G.add_node(tail.word)
#     G.add_edge(head.word, tail.word, label=relation)

# # 엣지 라벨을 포함한 그래프 그리기
# pos = nx.spring_layout(G)
# plt.figure(figsize=(12, 8))

# nx.draw(
#     G,
#     pos,
#     with_labels=True,
#     node_size=3000,
#     node_color="lightblue",
#     font_size=10,
#     font_weight="bold",
#     arrows=False,
#     edge_color="gray",
#     width=2,
#     font_family=fontprop.get_name(),
# )
# # nx.draw 인자 설명
# # pos: node의 위치를 지정하는 dict. pos[node]는 node의 위치를 나타내는 좌표
# # nx.spring_layout(G)을 사용하여 node의 위치를 설정했습니다. 이 레이아웃 알고리즘은 물리적 스프링 모델을 기반으로 하여 node를 화면에 적절히 배치합니다.
# # with_labels: node label 표시 여부
# # node_size: node 크기
# # arrows: edge 화살표 표시 여부
# # width: edge 두께
# # font_family=fontprop.get_name(): node label에 사용할 글꼴 지정

# # 엣지 라벨 추가
# edge_labels = {(head.word, tail.word): relation for head, tail, relation in data}
# nx.draw_networkx_edge_labels(
#     G, pos, edge_labels=edge_labels, font_color="red", font_family=fontprop.get_name()
# )

# plt.title("Knowledge Graph", fontproperties=fontprop)
# plt.show()
