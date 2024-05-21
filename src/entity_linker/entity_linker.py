from logging import getLogger, basicConfig
import pandas as pd
from knowledgebase.knowledgebase import Knowledgebase
from src.entity.entity import Entity
import lightning as L

logger = getLogger(__name__)

basicConfig(level="INFO", format="%(asctime)s - %(message)s")


knowledgebases = {
    "encykorea": "src/knowledge_base/한국학중앙연구원_한국민족문화대백과사전_20240130.csv",
}


class Hodu:
    """
    Entity Linker class for linking entities to the knowledge base. This is pipeline
    class for entity linking. It supprots knowledgebase loading, candidate generation,
    and candidate classification.
    """

    def __init__(self, knowledgebase: Knowledgebase, candidate_classifier=None):
        # Initialize Knowledge base
        # Health Check for the knowledge base connection.
        self.knowledgebase = knowledgebase
        self.candidate_classifier = candidate_classifier

    def _generate_candidates(self, entity: Entity):
        """
        Generate candidates for entity linking, for a single entity.

        Args:
            entity (_type_): _description_
        """
        return self.knowledgebase.generate_candidates(entity)

    def _get_candidate_scores(self, entity, candidates):
        """
        Get the scores for the candidates.

        Args:
            entity (_type_): _description_
            candidates (_type_): _description_

        Returns:
            _type_: _description_
        """
        pass

    def get_id(self, entity):
        """
        Get the ID of the entity from the knowledge base.

        Args:
            entity (_type_): _description_

        Returns:
            _type_: _description_
        """

        candidates = self._generate_candidates(entity)
        if len(candidates) < 1:
            return None

        if len(candidates) == 1:
            return candidates[0]

        return candidates[0]
        # candidate_scores = self._get_candidate_scores(entity, candidates)
