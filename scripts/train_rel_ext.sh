#!/bin/bash

python3 train.py --project "KingKorre" --log-model "all" --rel2id-path "./gpt_relationships_only_person.json" --max-epochs 25 --train-json-path "sample_data/all.json" --valid-json-path "sample_data/태조7월.json" --batch-size 48 --max-len 512 --train-val-split 0.8 --devices 0 &
p1=$!

python3 train.py --project "KingKorre" --log-model "all" --rel2id-path "./gpt_relationships_only_person.json" --max-epochs 25 --train-json-path "sample_data/all.json" --valid-json-path "sample_data/태조7월.json" --batch-size 48 --max-len 512 --train-val-split 0.8 --devices 1 &
p2=$!

python3 train.py --project "KingKorre" --log-model "all" --rel2id-path "./gpt_relationships_only_person.json" --max-epochs 25 --train-json-path "sample_data/all.json" --valid-json-path "sample_data/태조7월.json" --batch-size 48 --max-len 512 --train-val-split 0.8 --devices 2 &
p3=$!

python3 train.py --project "KingKorre" --log-model "all" --rel2id-path "./gpt_relationships_only_person.json" --max-epochs 25 --train-json-path "sample_data/all.json" --valid-json-path "sample_data/태조7월.json" --batch-size 48 --max-len 512 --train-val-split 0.8 --devices 3 &
p4=$!

wait $p1 $p2 $p3 $p4
