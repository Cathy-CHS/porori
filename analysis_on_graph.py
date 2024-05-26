import os, sys
import json

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import visual


def main():
    entities_json = "results/1597/linked_entity.json"
    triplets_csv = "results/1597/relationships.csv"

    visual.analyze_graph(
        entities_json,
        triplets_csv,
        export_dir="1597_graph_analysis_5",
        king_name="선조",
        degree_threshold=5,
    )


if __name__ == "__main__":
    main()
