from logging import getLogger, basicConfig
import pandas as pd

logger = getLogger(__name__)

basicConfig(level="INFO", format="%(asctime)s - %(message)s")


knowledgebases = {
    "encykorea": "src/knowledge_base/한국학중앙연구원_한국민족문화대백과사전_20240130.csv",
}


class Hodu:
    """
    Entity Linker class for linking entities to the knowledge base.
    """

    def __init__(self, knowledgebase="encykorea"):
        # Initialize Knowledge base
        # Health Check for the knowledge base connection.
        if knowledgebase not in knowledgebases:
            raise ValueError(f"Knowledge base {knowledgebase} not found.")

        self.knowledgebase_name = knowledgebase
        self.knowledgebase_data = None

        self.load_knowledgebase()

    def load_knowledgebase(self):
        """
        Load the knowledge base from the knowledge base path.

        Returns:
            _type_: _description_
        """
        knowledgebase_path = knowledgebases[self.knowledgebase_name]
        logger.info("Loading knowledge base from %s", knowledgebase_path)
        self.knowledgebase_data = pd.read_csv(knowledgebase_path)
        return pd.read_csv(knowledgebase_path)

    def _generate_candidates(self, entity):
        """
        Generate candidates for entity linking, for a single entity.

        Args:
            entity (_type_): _description_
        """
        pass

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

        candidate_scores = self._get_candidate_scores(entity, candidates)
