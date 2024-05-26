import os
import requests
from src.entity.entity import Entity
from typing import List

AVAILABLE_KNOWLEDGE_BASES = ["encykorea-api"]  # TODO: Add encykorea-local


class KnowledgeBaseEntity:
    def __init__(self, knowledgebase, name, entity_id):
        self.knowledgebase = knowledgebase
        self.name = name
        self.entity_id = entity_id

        self.definition = None
        self.summary = None
        self.description = None

        # self.get_context()

    def __str__(self):
        return f"{self.name} ({self.knowledgebase})"

    def get_context(self):
        """
        Get the candidate context of the entity from the knowledge base.

        Returns:
            dict: Context of the entity.
            Contains name(string), definition(string), and description(string).
        """
        pass


class EncyKoreaAPIEntity(KnowledgeBaseEntity):
    """
    EncyKorea API Entity class for linking.

    openAPI URLs (https://encykorea.aks.ac.kr/Guide/OpenApiUse)
        특정 항목 내용(항목ID는 항목 리스트에서 확인 가능)
        GET https://suny.aks.ac.kr:5143/api/Article/{항목ID}

    """

    def __init__(self, name, entity_id, access_key=None):
        """
        Initialize the EncyKorea API Entity. You should have an access key to use
        the API. Feed the access key as an argument or set it as an environment
        variable, "ENCYKOREA_API_KEY".

        Args:
            name (string): Name of the entity.
            entity_id (string): ID of the entity in the EncyKorea API.
            access_key (string, optional): The access key to the encykorea openapi.
            Defaults to None. If not provided, it will try to get the key from the
            environment variable.

        Raises:
            ValueError: If the access key is not provided or found in the environment.
        """
        super().__init__("encykorea-api", name, entity_id)

        if access_key is None:
            try:
                access_key = os.getenv("ENCYKOREA_API_KEY")
            except Exception as e:
                raise ValueError(
                    "Access key for EncyKorea API not provided or found in the environment."
                ) from e

        self.access_key = access_key
        self._get_content()

    def get_context(self):
        """
        Get the candidate context of the entity from the knowledge base.

        Returns:
            dict: Context of the entity.
            Contains name(string), definition(string), and description(string).
        """
        if not self.definition or not self.description:
            self._get_content()

        return {
            "name": self.name,
            "definition": self.definition,
            "description": self.description,
        }

    def _get_content(self):
        """
        Get definition and summary from the EncyKorea API, and update the entity.
        """
        url = f"https://suny.aks.ac.kr:5143/api/Article/{self.entity_id}"
        headers = {"accessKey": self.access_key}
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code != 200:
            raise ValueError(
                f"Failed to get content for {self.name} from EncyKorea API."
            )

        data = response.json()["article"]
        self.name = data["headword"]
        self.definition = data["definition"]
        self.description = data["body"]

        summary = data["summary"]
        if summary:
            self.summary = summary


class Knowledgebase:
    def __init__(self, knowledgebase="encykorea-api", max_candidates=5):
        """
        Initialize the knowledge base.

        Args:
            knowledgebase (str, optional): Type of knowledgebase. Defaults to "encykorea-api".
            max_candidates (int, optional): The maximum number of candidate entities. Defaults to 5.

        Raises:
            ValueError: If the knowledge base is not found in the available knowledge bases.
        """

        if knowledgebase not in AVAILABLE_KNOWLEDGE_BASES:
            raise ValueError(f"Knowledge base {knowledgebase} not found.")

        self.knowledgebase_name = knowledgebase
        self.max_candidates = max_candidates

    def generate_candidates(self, entity: Entity):
        """
        Generate candidates for entity linking from the knowledge base.

        Args:
            entity (Entity): Detected entity from the text. Need to be linked to an
            entity in the knowledgebase.

        Returns:
            [Entity]:
        """
        pass


class EncyKoreaAPIKnowledgeBase(Knowledgebase):
    def __init__(self, max_candidates=5, access_key=None):
        super().__init__("encykorea-api")

        if access_key is None:
            try:
                access_key = os.getenv("ENCYKOREA_API_KEY")
            except Exception as e:
                raise ValueError(
                    "Access key for EncyKorea API not provided or found in the environment."
                ) from e

        self.access_key = access_key

    def generate_candidates(self, entity: Entity) -> List[EncyKoreaAPIEntity]:
        """
        Generate candidates for entity linking from the EncyKorea API.

        항목 검색
        GET https://suny.aks.ac.kr:5143/api/Article/Search/{검색어}?page={페이지}&field={분야}&type={유형}
        Args:
            entity (Entity): Detected entity from the text. Need to be linked to an
            entity in the knowledgebase.

        Returns:
            List[EncyKoreaAPIEntity]: List of candidate entities.
        """
        url = f"https://suny.aks.ac.kr:5143/api/Article/Search/{entity.word}?page=1"

        headers = {"accessKey": self.access_key}
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code != 200:
            raise ValueError(
                f"Failed to get candidates for {entity.word} from EncyKorea API."
            )

        data = response.json()
        entities = []

        for item in data["articles"]:
            entity = EncyKoreaAPIEntity(
                name=item["headword"], entity_id=item["eid"], access_key=self.access_key
            )
            entities.append(entity)
            if len(entities) >= self.max_candidates:
                break
        # print(entities)
        return entities
