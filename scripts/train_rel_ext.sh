python3 train.py --project "KingKorre" --log-model "all" --rel2id-path "./gpt_relationships_only_person.json" --max-epochs 25 --train-json-path "sample_data/all.json" \
--valid-json-path "sample_data/태조7월.json" --batch-size 24 --max-len 512 --train-val-split 0.8


# python3 train.py --project "KingKorre" --log-model "all" --rel2id-path "./gpt_relationships_only_person.json" --max-epochs 25 --train-json-path "sample_data/all.json" \
# --valid-json-path "sample_data/태조7월.json" --batch-size 24 --max-len 512  --pooling-mode "max"

python3 train.py --project "KingKorre" --log-model "all" --rel2id-path "./gpt_relationships_only_person.json" --max-epochs 25 --train-json-path "sample_data/all.json" \
--valid-json-path "sample_data/태조7월.json" --batch-size 24 --max-len 512  --pooling-mode "mean" --train-val-split 0.8
