python3 train.py --project "KingKorre" --log-model "all" --rel2id-path "./gpt_relationships_only_person.json" --max-epochs 20 --train-json-path "sample_data/sejong_jungjong.json" \
--valid-json-path "sample_data/태조7월.json" --batch-size 24 --max-len 512 


python3 train.py --project "KingKorre" --log-model "all" --rel2id-path "./gpt_relationships_only_person.json" --max-epochs 20 --train-json-path "sample_data/sejong_jungjong.json" \
--valid-json-path "sample_data/태조7월.json" --batch-size 24 --max-len 512  --pooling-mode "max"

python3 train.py --project "KingKorre" --log-model "all" --rel2id-path "./gpt_relationships_only_person.json" --max-epochs 20 --train-json-path "sample_data/sejong_jungjong.json" \
--valid-json-path "sample_data/태조7월.json" --batch-size 24 --max-len 512  --pooling-mode "mean"
