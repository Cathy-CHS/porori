from typing import List, Union
import json


class Entity:
    def __init__(self, entity: str, word: str, start: int, end: int, uri=None) -> None:

        self.entity = entity
        self.word = word
        self.start = start
        self.end = end
        self.uri = uri
        self.items = []

    def __str__(self) -> str:
        return f"Entity: {self.entity}, Word: {self.word}, Start: {self.start}, End: {self.end}"


class Linked_Entity:
    def __init__(self, name, entity_id):
        self.name = name
        self.entity_id = entity_id
        self.items = []

    def add_item(self, start, end):
        self.items.append((start, end))

    def __str__(self) -> str:
        return f"Entity: {self.name}, Word: {self.entity_id}"