from transformers import pipeline

class Dotori:
    def __init__(self):
        # load Roberta Entity Recognition model
        # Use a pipeline as a high-level helper
    
        pipe = pipeline("token-classification", model="yongsun-yoon/klue-roberta-base-ner")

    def extract_entities(self, text):
        entities = self.pipe(text)
        return entities