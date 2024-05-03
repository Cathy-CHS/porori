import re
from transformers import AutoTokenizer, pipeline
from src.entity import Entity
class Dotori:
    def __init__(self,  max_tokens=1000):
        # load Roberta Entity Recognition model
        # Use a pipeline as a high-level helper
        self.pipe = pipeline("token-classification", model="yongsun-yoon/klue-roberta-base-ner")
        self.tokenizer = AutoTokenizer.from_pretrained("yongsun-yoon/klue-roberta-base-ner")
        self.max_tokens = max_tokens
        self.processed_entities = []

    def extract_entities(self, text):
        # Split text into sentences using regex
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        entities = []
        for sentence in sentences:
            sentence_entities = self.pipe(sentence)
            entities.extend(sentence_entities)

        return entities
        
    def _extract_entities(self, text):
        # Split text into sentences using regex
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Tokenize each sentence
        tokenized_sentences = [self.tokenizer.tokenize(sentence) for sentence in sentences]
        
        # Combine tokenized sentences into sequences
        start_index_per_sequence = [0]
        sequences = []
        current_sequence = []
        current_sentence_length = 0
        for i, sentence in enumerate(tokenized_sentences):
            if len(current_sequence) + len(sentence) <= self.max_tokens:
                current_sequence.extend(sentence)
                current_sentence_length += len(sentences[i])
            else:
                sequences.append(current_sequence)
                start_index_per_sequence.append(current_sentence_length)
                
                current_sequence = sentence
                current_sentence_length += len(sentences[i])
                
        if current_sequence:
            sequences.append(current_sequence)
            start_index_per_sequence.append(current_sentence_length)
        
        # Perform entity recognition on each sequence
        entities = []
        
        for i, sequence in enumerate(sequences):
            sequence_text = self.tokenizer.convert_tokens_to_string(sequence)
            sequence_entities = self.pipe(sequence_text)
            start_index = start_index_per_sequence[i]
            new_entities = []
            for entity in sequence_entities:
                e_new = {
                    'entity': entity['entity'],
                    'score': entity['score'],
                    'start': entity['start'] + start_index,
                    'end': entity['end'] + start_index,
                    'word': entity['word']
                }
                new_entities.append(e_new)

            entities.extend(new_entities)

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
        print(e)
        out.write(str(e)+"\n")
    out.close
    # /home/codespace/.python/current/bin/python3 /workspaces/porori/entity_extractor.py