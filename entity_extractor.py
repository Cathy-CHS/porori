from transformers import pipeline
from spacy.pipeline.entity_linker import DEFAULT_NEL_MODEL
import spacy
class Dotori:
    def __init__(self):
        # load Roberta Entity Recognition model
        # Use a pipeline as a high-level helper
        self.pipe = pipeline("token-classification", model="yongsun-yoon/klue-roberta-base-ner")
        self.processed_entities = []
        

        spacy.prefer_gpu()
        # self.nlp = spacy.load("ko_core_web_sm")
        
        # config = {
        # "labels_discard": [],
        # "n_sents": 0,
        # "incl_prior": True,
        # "incl_context": True,
        # "model": DEFAULT_NEL_MODEL,
        # "entity_vector_length": 64,
        # "get_candidates": {'@misc': 'spacy.CandidateGenerator.v1'},
        # "threshold": None,
        # }
        # self.nlp.add_pipe("entity_linker", config=config)

    def extract_entities(self, text):
        entities = self.pipe(text)
        return entities

    def group_chunk(self, entities):
        # I로 시작하는 같은 entity type의 chunk 묶기
        current_group = None

        for entity in entities:
            if entity['entity'].startswith('B-'):
                if current_group:
                    self.processed_entities.append(current_group)
                current_group = {
                    'entity': entity['entity'][2:],  # Remove 'B-' prefix
                    'word': entity['word'].replace('##', ''),  # Clean up word
                    'start': entity['start'],
                    'end': entity['end']
                }
            elif entity['entity'].startswith('I-') and current_group:
                if entity['entity'][2:] == current_group['entity']:  # Check if the same entity type
                    current_group['word'] += entity['word'].replace('##', '')
                    current_group['end'] = entity['end']
                else:  # Different entity type
                    if current_group:
                        self.processed_entities.append(current_group)
                    current_group = {
                        'entity': entity['entity'][2:],  # Remove 'I-' prefix
                        'word': entity['word'].replace('##', ''),  # Clean up word
                        'start': entity['start'],
                        'end': entity['end']
                    }
                    # current_group = None  # Reset for safety
            print("current_group: ", current_group)

        if current_group:
            self.processed_entities.append(current_group)

        return self.processed_entities
                

if __name__ == "__main__":
    f = open("output.txt", 'r')
    # text = """
    #     태조가 수창궁에서 왕위에 올랐다. 이보다 먼저 이달 12일에 공양왕이 장차 태조의 사제로 거둥하여 술자리를 베풀고 태조와 더불어 동맹하려고 하여 의장이 이미 늘어섰는데, 시중 배극렴 등이 왕대비에게 아뢰었다.지금 왕이 혼암하여 임금의 도리를 이미 잃고 인심도 이미 떠나갔으므로, 사직과 백성의 주재자가 될 수 없으니 이를 폐하기를 청합니다.
    #     """
    text = f.read()
    dotori = Dotori()
    entities = dotori.extract_entities(text)
    result = dotori.group_chunk(entities)
    for e in entities:
        print(e)
    print('---------------------------------')
    for e in result:
        print(e)
    # /home/codespace/.python/current/bin/python3 /workspaces/porori/entity_extractor.py