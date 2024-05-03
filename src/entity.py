from typing import List, Union

class Entity:
    def __init__(self,
                    entity: str,
                    word: str,
                    start: int,
                    end: int,
                    uri = None) -> None:

        self.entity = entity
        self.word = word
        self.start = start
        self.end = end
        self.uri = uri

    def __str__(self) -> str:
        return f"Entity: {self.entity}, Word: {self.word}, Start: {self.start}, End: {self.end}"