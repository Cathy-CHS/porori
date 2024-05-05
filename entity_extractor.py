import re
from transformers import AutoTokenizer, pipeline
from src.entity import Entity
class Dotori:
    def __init__(self, max_tokens=512):
        # load Roberta Entity Recognition model
        # Use a pipeline as a high-level helper
        self.pipe = pipeline("token-classification", model="yongsun-yoon/klue-roberta-base-ner")
        self.tokenizer = AutoTokenizer.from_pretrained("yongsun-yoon/klue-roberta-base-ner")
        self.max_tokens = max_tokens
        self.processed_entities = []

    def extract_entities(self, text):
        # 문장으로 나누기
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # token화 하기 전 원본 문장의 길이를 저장하기 위한 변수
        start_positions = []
        end_positions = []
        offset = 0
        
        # 문장의 시작, 끝 위치 계산
        for sentence in sentences:
            start_positions.append(offset)
            end_positions.append(offset + len(sentence))
            offset += len(sentence) + 1 

        entities = []
        for start, end in zip(start_positions, end_positions):
            sentence_text = text[start:end]
            sentence_entities = self.pipe(sentence_text)
            for entity in sentence_entities:
                # 시작, 끝 위치 조정
                entity['start'] += start
                entity['end'] += start
                entities.append(entity)
        
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

        if current_group:
            self.processed_entities.append(current_group)

        return self.processed_entities
    

if __name__ == "__main__":
    out = open("entities_output.txt", 'w', encoding='utf-8')
    f = open("output.txt", 'r', encoding='utf-8')
    
    text = f.read()
    dotori = Dotori()
    entities = dotori.extract_entities(text)
    result = dotori.group_chunk(entities)
    
    for e in entities:
        print(e)
        
    print('---------------------------------')
    
    for e in result:
        # print(e)
        out.write(str(e)+"\n")
    out.close()