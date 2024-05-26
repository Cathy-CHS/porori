import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx
import os
import csv
import json
from relationship.relationship_extractor import load_linked_entities_from_json
import pandas as pd
import numpy as np

# import csv
# from entity.entity import Entity

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "NanumGothic"


def read_triplets_from_csv(triplets_csv):
    triplets = pd.read_csv(triplets_csv)
    return [
        (row["head_entity_id"], row["tail_entity_id"], row["relation"])
        for _, row in triplets.iterrows()
    ]


def load_graph(entities_json: str, triplets_csv: str) -> nx.DiGraph:
    # read entities

    linked_entities = load_linked_entities_from_json(entities_json)

    # read triplets
    triplets = read_triplets_from_csv(triplets_csv)

    # construct networkx directed graph with edge attributes
    G = nx.DiGraph()
    for entity in linked_entities:
        G.add_node(entity.entity_id, name=entity.name)

    for head, tail, relation in triplets:
        G.add_edge(head, tail, relationship=relation)
    return G


def plot_communities(G, communities, title, export_path=None):
    pos = nx.spring_layout(G, k=0.3, iterations=25)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(communities)))

    print(f"Drawing graph for {title} communities...")
    plt.figure(figsize=(15, 15))
    for color, community in zip(colors, communities):
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=community,
            node_color=[color],
            label=f"Community {communities.index(community)}",
            node_size=100,
        )
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)

    labels = {node: data["name"] for node, data in G.nodes(data=True)}

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=labels)
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig(export_path)
    plt.close()


def process_graph(g: nx.DiGraph, export_dir: str, king_name=None):
    """
    1. remove nodes with degree < 3
    2. remove isolated nodes

    -> results


    """
    ### PREPROCESSING ###
    # 1. remove nodes with degree < 3
    nodes_to_remove = [node for node in g.nodes if g.degree(node) < 3]
    g.remove_nodes_from(nodes_to_remove)
    print(f"Removed {len(nodes_to_remove)} nodes with degree < 3")
    # 2. remove isolated nodes
    isolated_nodes = list(nx.isolates(g))
    g.remove_nodes_from(isolated_nodes)
    print(f"Removed {len(isolated_nodes)} isolated nodes")
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        print(f"Created directory: {export_dir}")
    nx.write_gexf(g, os.path.join(export_dir, "processed_graph.gexf"))
    print(f"Exported processed graph to {export_dir}/processed_graph.gexf")
    #### RESULTS ####
    print("============= Graph Analysis =============")
    print(f"Number of nodes: {g.number_of_nodes()}")
    print(f"Number of edges: {g.number_of_edges()}")
    print(f"average degree of the graph: {np.mean([d for n, d in g.degree()])}")

    # save graph fig
    visualize_graph(g, output_dir=export_dir, siloc_title=king_name)
    # 1. pagerank
    statistics_df = pd.DataFrame()

    print("============= PageRank =============")
    pagerank = nx.pagerank(g)
    sorted_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 nodes by pagerank:")
    for node, pr in sorted_pagerank[:10]:
        print(f"{g.nodes[node]['name']} (id: {node}) - pagerank: {pr:.4f}")

    # plot sorted pagerank with plt.plot
    plt.figure(figsize=(10, 7))
    plt.plot([x[1] for x in sorted_pagerank])
    plt.title("PageRank")
    plt.savefig(f"{export_dir}/pagerank.png")
    plt.close()

    # 2. degree centrality
    print("============= Degree Centrality =============")
    plt.figure(figsize=(10, 7))
    degree_centrality = nx.degree_centrality(g)
    betweenness_centrality = nx.betweenness_centrality(g)
    closeness_centrality = nx.closeness_centrality(g)
    sorted_degree_centrality = sorted(
        degree_centrality.items(), key=lambda x: x[1], reverse=True
    )

    print("Top 10 nodes by degree centrality:")
    for node, dc in sorted_degree_centrality[:10]:
        print(f"{g.nodes[node]['name']} (id: {node}) - degree centrality: {dc:.4f}")
    sorted_betweenness_centrality = sorted(
        betweenness_centrality.items(), key=lambda x: x[1], reverse=True
    )

    # plot sorted betweenness centrality with plt.plot
    print("============= Betweenness Centrality =============")
    plt.figure(figsize=(10, 7))
    plt.plot([x[1] for x in sorted_betweenness_centrality])
    plt.title("Betweenness Centrality")
    plt.savefig(f"{export_dir}/betweenness_centrality.png")
    plt.close()

    print("Top 10 nodes by betweenness centrality:")
    for node, bc in sorted_betweenness_centrality[:10]:
        print(
            f"{g.nodes[node]['name']} (id: {node}) - betweenness centrality: {bc:.4f}"
        )
    sorted_closeness_centrality = sorted(
        closeness_centrality.items(), key=lambda x: x[1], reverse=True
    )
    print("Top 10 nodes by closeness centrality:")
    for node, cc in sorted_closeness_centrality[:10]:
        print(f"{g.nodes[node]['name']} (id: {node}) - closeness centrality: {cc:.4f}")

    # plot sorted closeness centrality with plt.plot
    print("============= Closeness Centrality =============")

    for node, cc in sorted_closeness_centrality[:10]:
        print(f"{g.nodes[node]['name']} (id: {node}) - closeness centrality: {cc:.4f}")

    plt.figure(figsize=(10, 7))
    plt.plot([x[1] for x in sorted_closeness_centrality])
    plt.title("Closeness Centrality")
    plt.savefig(f"{export_dir}/closeness_centrality.png")
    plt.close()

    # 3. community detection
    print("============= Community Detection =============")
    # 3-1. greedy modularity communities
    gr_communities = list(nx.algorithms.community.greedy_modularity_communities(g))
    print(f"Number of communities: {len(gr_communities)}")
    gr_communities = [list(community) for community in gr_communities]
    plot_communities(
        g,
        gr_communities,
        "Greedy Modularity Communities",
        os.path.join(export_dir, "greedy_modularity_communities.png"),
    )

    # # 3-2 girvan-newman communities
    # gn_communities = list(nx.algorithms.community.girvan_newman(g))
    # print(f"Number of communities: {len(gn_communities)}")
    # gn_communities = [list(community) for community in gn_communities]
    # plot_communities(
    #     g,
    #     gn_communities,
    #     "Girvan-Newman Communities",
    #     os.path.join(export_dir, "girvan_newman_communities.png"),
    # )

    # 4 Hubs & Authority
    print("============= Hubs & Authority =============\n")
    hubs, authorities = nx.hits(g)
    sorted_hubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)
    sorted_authorities = sorted(authorities.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 nodes by hubs:")
    print("----------------------")
    for node, hub in sorted_hubs[:10]:
        print(f"{g.nodes[node]['name']} (id: {node}) - hubs: {hub:.4f}")
    print("----------------------")
    print("Top 10 nodes by authorities:")
    print("----------------------")
    for node, authority in sorted_authorities[:10]:
        print(f"{g.nodes[node]['name']} (id: {node}) - authorities: {authority:.4f}")
    print("----------------------")
    # 5. plot ppr with king
    print("============= Personalized PageRank among King =============")
    if king_name is not None:
        king_id = None
        for node in g.nodes:
            if g.nodes[node]["name"] == king_name:
                king_id = node
                break
        if king_id is None:
            print(f"King {king_name} not found in the graph")
        else:
            print(f"Found node id of the king {king_name}")
            personlaization = {node: 0 for node in g.nodes()}
            personlaization[king_id] = 1
            ppr_score = nx.pagerank(g, personalization=personlaization)
            sorted_ppr = sorted(ppr_score.items(), key=lambda x: x[1], reverse=True)
            print("Top 10 nodes by PPR with the king:")
            for node, ppr in sorted_ppr[:10]:
                print(f"{g.nodes[node]['name']} (id: {node}) - PPR: {ppr:.4f}")

    df_dicts = []
    for node in g.nodes:
        # print(node, g.nodes[node]["name"])
        df_dicts.append(
            {
                "entity_id": node,
                "entity_name": g.nodes[node]["name"],
                "pagerank": pagerank[node],
                "degree_centrality": degree_centrality[node],
                "betweenness_centrality": betweenness_centrality[node],
                "closeness_centrality": closeness_centrality[node],
                "hubs": hubs[node],
                "authorities": authorities[node],
                "ppr_king": ppr_score[node] if king_id is not None else None,
            }
        )
    statistics_df = pd.DataFrame(df_dicts)
    statistics_df.to_csv(
        os.path.join(export_dir, "node_statistics.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def analyze_graph(
    entities_json: str, triplets_csv: str, export_dir: str, king_name=None
):
    G = load_graph(entities_json, triplets_csv)
    process_graph(G, export_dir, king_name)


def plot_king_ppr(G, king_id, alpha=0.85):
    pass


def visualize_graph(G, output_dir="output", siloc_title="1597 Sunjo"):
    # font_path = os.path.join("fonts", "NANUMGOTHIC.TTF")  # 한국어 폰트 파일 경로
    # fontprop = fm.FontProperties(fname=font_path)
    # plt.rc("font", family=fontprop.get_name())

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
    edge_labels = {
        (head, tail): relation for head, tail, relation in G.edges(data="relationship")
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_color="red",
        # font_family=fontprop.get_name(),
    )

    plt.title(
        f"Knowledge Graph for {siloc_title}",
        #   fontproperties=fontprop
    )
    plt.savefig(
        os.path.join(output_dir, f"{siloc_title}_knowledgegraph.png"),
        dpi=300,
        facecolor="white",
        edgecolor="black",
        orientation="portrait",
        format="png",
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.1,
    )

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
