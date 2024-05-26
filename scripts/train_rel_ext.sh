#!/bin/bash

python3 train.py --project "KingKorre" --log-model "all" --rel2id-path "./gpt_relationships_only_person.json" --max-epochs 30 --train-json-path "sample_data/all.json" --valid-json-path "sample_data/test_dataset.json" --batch-size 48 --max-len 512  --devices 0 


