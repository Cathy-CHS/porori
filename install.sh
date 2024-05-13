pip install -r requirements.txt
pip install -U pip setuptools wheel
pip install -U 'spacy[cuda12x,transformers,lookups]'
python -m spacy download en_core_web_sm
python -m spacy download ko_core_news_sm
playwright install