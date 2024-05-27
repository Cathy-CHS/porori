import os, sys
import json

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import visual
from fire import Fire


def main(
    graph_dir="results/sunjo_1587/",
    export_dir="graph_analysis_results/sunjo_1587",
    king_name="선조",
    degree_threshold=5,
):
    entities_json = os.path.join(graph_dir, "linked_entity.json")
    triplets_csv = os.path.join(graph_dir, "relationships.csv")

    visual.analyze_graph(
        entities_json,
        triplets_csv,
        export_dir=export_dir,
        king_name=king_name,
        degree_threshold=degree_threshold,
    )


if __name__ == "__main__":
    Fire(main)
