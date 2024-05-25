# -*- coding: utf-8 -*-
import re
from multiprocessing import Pool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import uuid
from typing import Dict, List, TypedDict, Optional
from langchain.callbacks import get_openai_callback
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
import json

from dotenv import load_dotenv
import os

load_dotenv()
# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)


# Define data models


class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the person.")


class Relationship(BaseModel):
    """Relationship between two people."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    head: Optional[str] = Field(
        default=None, description="The person in the head of the relationship."
    )
    tail: Optional[str] = Field(
        default=None, description="The person in the tail of the relationship."
    )
    relationship: Optional[str] = Field(
        default=None, description="The relationship between the two people."
    )


class Entities(BaseModel):
    """list of people."""

    entities: List[str] = Field(default=None, description="The list of the people.")


class Data(BaseModel):
    """Extracted data about people."""

    # Creates a model so that we can extract multiple entities.
    # data: Dict["entities_list":Entities, "relationship_list":Relationship]
    entities_list: List[str] = Field(default=[], description="List of people.")
    relationship_list: List[Relationship] = Field(
        default=[], description="List of relationships between the entities."
    )


# Define Examples


class Example(TypedDict):
    """A representation of an example consisting of text input and expected tool calls.

    For extraction, the tool calls are represented as instances of pydantic model.
    """

    input: str  # This is the example text
    tool_calls: List[Data]  # Instances of pydantic model that should be extracted


def tool_example_to_messages(example: Example) -> List[BaseMessage]:
    """Convert an example into a list of messages that can be fed into an LLM.

    This code is an adapter that converts our example to a list of messages
    that can be fed into a chat model.

    The list of messages per example corresponds to:

    1) HumanMessage: contains the content from which content should be extracted.
    2) AIMessage: contains the extracted information from the model
    3) ToolMessage: contains confirmation to the model that the model requested a tool correctly.

    The ToolMessage is required because some of the chat models are hyper-optimized for agents
    rather than for an extraction use case.
    """

    messages: List[BaseMessage] = [HumanMessage(content="input: " + example["input"])]

    openai_tool_calls = []
    for data in example["tool_calls"]:
        openai_tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    # The name of the function right now corresponds
                    # to the name of the pydantic model
                    # This is implicit in the API right now,
                    # and will be improved over time.
                    "name": "Data",
                    "arguments": data.json(),
                },
            }
        )
    messages.append(
        AIMessage(content="", additional_kwargs={"tool_calls": openai_tool_calls})
    )
    tool_outputs = example.get("tool_outputs") or [
        "You have correctly called this tool."
    ] * len(openai_tool_calls)
    for output, tool_call in zip(tool_outputs, openai_tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))

    return messages


def export_data_list(
    data_list: List[Data],
    res: Dict[str, List],
    input_batches: List[str] = None,
):
    for i, data in enumerate(data_list):
        dict_data = {
            "entities": data.entities_list,
            "relationships": [
                (rel.head, rel.tail, rel.relationship) for rel in data.relationship_list
            ],
            "input_text": input_batches[i] if input_batches else None,
        }
        res["data"].append(dict_data)
    
    return res

def export_data_list_as_json(res, output_file):
    with open(output_file, "w", encoding="UTF-8-sig") as f:
        json.dump(
            res,
            f,
            ensure_ascii=False,
            indent=4,
        )


def get_relationships_from_text(llm, texts: List[str]) -> Data:
    examples = [
        (
            """
            태조가 수창궁에서 왕위에 올랐다. 이보다 먼저 이달 12일에 공양왕이 장차 태조의 사제로 거둥하여 술자리를 베풀고 태조와 더불어 동맹하려고 하여 의장이 이미 늘어섰는데, 시중 배극렴 등이 왕대비에게 아뢰었다."지금 왕이 혼암하여 임금의 도리를 이미 잃고 인심도 이미 떠나갔으므로, 사직과 백성의 주재자가 될 수 없으니 이를 폐하기를 청합니다."마침내 왕대비의 교지를 받들어 공양왕을 폐하기로 일이 이미 결정되었는데, 남은이 드디어 문하 평리 정희계와 함께 교지를 가지고 북천동의 시좌궁 에 이르러 교지를 선포하니, 공양왕이 부복하고 명령을 듣고 말하기를,"내가 본디 임금이 되고 싶지 않았는데 여러 신하들이 나를 강제로 왕으로 세웠습니다. 내가 성품이 불민하여 사기를 알지 못하니 어찌 신하의 심정을 거스린 일이 없겠습니까"하면서, 이내 울어 눈물이 두서너 줄기 흘러내리었다. 마침내 왕위를 물려주고 원주로 가니, 백관이 국새를 받들어 왕대비전에 두고 모든 정무를 나아가 품명하여 재결하였다. 13일임진에 대비가 교지를 선포하여 태조를 감록국사로 삼았다.
            """,
            Data(
                entities_list=["태조", "공양왕", "배극렴", "왕대비", "남은", "정희계"],
                relationship_list=[
                    Relationship(head="태조", tail="공양왕", relationship="L026"),
                    Relationship(head="공양왕", tail="태조", relationship="L027"),
                    Relationship(head="배극렴", tail="왕대비", relationship="L028"),
                    Relationship(head="남은", tail="공양왕", relationship="L024"),
                    Relationship(head="정희계", tail="공양왕", relationship="L024"),
                    Relationship(head="왕대비", tail="공양왕", relationship="L024"),
                    Relationship(head="왕대비", tail="태조", relationship="L005"),
                ],
            ),
        ),
        (
            """
            이천우를 붙잡고 겨우 침문 밖으로 나오니 백관이 늘어서서 절하고 북을 치면서 만세를 불렀다. 태조가 매우 두려워하면서 스스로 용납할 곳이 없는 듯하니, 극렴 등이 합사하여 왕위에 오르기를 권고하였다."나라에 임금이 있는 것은 위로는 사직을 받들고 아래로는 백성을 편안하게 할 뿐입니다. “
            """,
            Data(
                entities_list=["이천우", "백관", "태조", "극렴"],
                relationship_list=[
                    Relationship(head="극렴", tail="태조", relationship="L024"),
                ],
            ),
        ),
        (
            """
            선공감에 명하여 양청의 보첨을 짓게 하니, 선공감에서 아뢰었다."옛날에는 조은도가 선공감에 소속되었으므로, 매년 가을에 갈대를 베어서 국가의 용도에 공급하였사오나, 지금은 급전사에서 과전에 이를 소속시켜 참찬문하부사 정희계에게 주었으니, 양청의 차양은 갈대로써 덮기는 어렵겠습니다."임금이 말하기를,"고려 왕조의 공양군이 이 토지를 사사로이 그 아들에게 주었으니 좋은 일이 아닌데, 지금 급전사에서 그대로 과전에 소속시켜서 국가의 용도는 돌보지도 않고 재상에게 아첨을 구하니 옳은 일이 아니다."하고는, 즉시 급전사 장무 이재를 순군옥에 내려 가두고, 선공감 승 박자량으로 하여금 답험하게 하였다.
            """,
            Data(
                entities_list=["정희계", "임금", "공양군", "이재", "박자량"],
                relationship_list=[
                    Relationship(head="임금", tail="공양군", relationship="L008"),
                    Relationship(head="임금", tail="이재", relationship="L013"),
                    Relationship(head="임금", tail="박자량", relationship="L024"),
                ],
            ),
        ),
        (
            """
            간관이 상소하였는데, 대략은 이러하였다."그윽이 보건대, 상의중추원사 유양이 전에 계림에 있을 때에, 왜구가 투항할 즈음을 당하여 단기로 가서 적을 보고 화복으로 달래어, 왜노로 하여금 자식을 볼모로 바치고 정성을 다하게 하였으니, 마땅히 더욱 부지런히 하고 게을리 하지 말아서 그 공을 이루어야할 터인데, 왜선이 와서 정박한 뒤에 병이 심하다는 핑계로 이해의 기미를 조금도 생각하지 않고, 중 의운을 보내어 의심이 나서 도망하여 돌아가게 하였고, 또 중이 왕래한 형적을 비밀히 하여 조정에 나타내어 고하려 하지 않고, 그 까닭을 핵문하여도 말을 놀리는 것이 교묘하고 거짓되어 사정을 나타내지 않으니, 자못 간사하고 속이는 데에 가깝습니다. 원컨대 유사로 하여금 직첩을 거두고 정상과 사유를 밝게 국문 하소서."임금이 단지 외방에 부처하고, 의운의 출현을 기다려 빙문하여 사실을 조사하게 하였다.
            """,
            Data(
                entities_list=["간관", "유양", "임금", "의운", "왜구"],
                relationship_list=[
                    Relationship(head="간관", tail="유양", relationship="L025"),
                    Relationship(head="임금", tail="유양", relationship="L024"),
                    Relationship(head="임금", tail="의운", relationship="L024"),
                ],
            ),
        ),
    ]

    messages = []

    for text, tool_call in examples:
        messages.extend(
            tool_example_to_messages({"input": text, "tool_calls": [tool_call]})
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert relationship extraction algorithm. "
                """
                Below is a list of relationships. Please refer to the above relationships and extract all existing relationships from the following text. Only consider person entities.
                (
                "L001": "Head가 Tail을 죽임. (Kills)",
                "L002": "Tail이 Head를 죽임. (Killed by)",
                "L003": "Head가 Tail을 공격함. (Attacks)",
                "L004": "Tail이 Head를 공격함. (Attacked by)",
                "L005": "Head가 Tail을 도와줌. (Helps)",
                "L006": "Head가 Tail에게 도움을 요청함. (Asks for help)",
                "L007": "Head가 Tail을 칭찬함. (Compliments)",
                "L008": "Head가 Tail을 비난함. (Blames)",
                "L009": "Head가 Tail을 믿음. (Trusts)",
                "L010": "Head가 Tail을 의심함. (Distrusts)",
                "L011": "Head가 Tail을 좋아함. (Likes)",
                "L012": "Head가 Tail을 싫어함. (Dislikes)",
                "L013": "Head가 Tail을 처벌시킴. (Punishes)",
                "L014": "Head가 Tail을 위로함. (Comforts)",
                "L015": "Head가 Tail을 무시함. (Ignores)",
                "L016": "Tail이 Head를 놀리거나 괴롭힘. (Teases)",
                "L017": "Head가 Tail의 가족임. (Family)",
                "L019": "Head가 Tail의 상관임. (Serves)",
                "L020": "Head가 Tail의 부하임.(Served by)",
                "L021": "Head가 Tail의 스승임 (Mentor)",
                "L022": "Tail이 Head의 스승임 (Mentored by)",
                "L023": "Head가 Tail에 대해 반발함. (Rebels against)",
                "L024": "Head가 Tail에게 지시함. (Commands)",
                "L025": "Head가 Tail을 고발함. (Accuses)",
                "L026": "Head가 Tail의 지위를 승계함 (Succeeds)",
                "L027": "Head가 Tail의 지위를 승계받음 (Succeeded by)",
                "L028": "Head가 Tail에게 보고함 (Reports to)",
                "P108": "해당 인물을 고용한 곳, 사람, 기관. (employer)",
                "P127": "해당 개체의 소유자 (owned by)",
                "P1365": "해당 개체는 다음으로부터 이어짐 (replaces)",
                "P1366": "해당 개체는 다음으로 이어짐 (replaced by)",
                "P137": "해당 개체의 운영 주체 (operator)",
                "P138": "해당 개체의 명칭은 다음에서 유래되었음 (named after)",
                "P155": "시간적 또는 공간적으로 앞에 오는 것 (follows)",
                "P156": "시간적 또는 공간적으로 뒤에 오는 것 (followed by)",
                "P1696": "반대 속성 (inverse property)",
                "P172": "다음은 해당 개체의 민족에 속함 (ethnic group)",
                "P176": "해당 개체의 제조사 (manufacturer)",
                "P178": "해당 개체를 개발한 주체 (developer)",
                "P1830": "해당 개체가 소유한 것 (owner of)",
                "P20": "사망한 곳 (place of death)",
                "P22": "해당 개체의 친아버지 (father)",
                "P26": "해당 개체의 배우자 (spouse)",
                "P2789": "해당 개체가 다음과 연결되어 있음 (connects with)",
                "P279": "하위 개념 (subclass of)",
                "P2936": "해당 장소나 사건에서 다음 언어가 주로 사용됨 (language used)",
                "P31": "다음 종류에 속함 (instance of)",
                "P3373": "해당 개체의 친형제자매 (sibling)",
                "P3730": "해당 개체의 바로 윗 계급 (next higher rank)",
                "P39": "해당 인물의 현재 또는 과거의 직위 (position held)",
                "P40": "해당 개체의 자녀 (child)",
                "P413": "해당 선수가 맡은 포지션 (position played on team / speciality)",
                "P463": "다음 공동체의 구성원임 (member of)",
                "P527": "해당 개체가 다음으로 이루어져 있음 (has part)",
                "P54": "해당 개체의 현재 또는 과거의 소속 팀 (member of sports team)",
                "P551": "해당 인물의 거주지 (residence)",
                "P6": "해당 정부 또는 지자체의 수장 (head of government)",
                "P607": "해당 개체가 참전한 전쟁 (conflict)",
                "P664": "주최자 (organizer)",
                "P69": "해당 인물의 학교 (educated at)",
                "P710": "해당 사건에 다음이 참여함 (participant)",
                "P737": "다음에게서 영향을 받음 (influenced by)",
                "P807": "해당 개체가 다음으로부터 갈라져 나와 시작된 것임 (separated from)",
                "P945": "해당 개체가 섬기는 국가나 세력 (allegiance)",
                "P97": "작위 (noble title)"
                )

                The output shape will be list of dictionaries like this.

                First generate the list of entities in the input text. Only find person entities. list your findings in this format:

                [entity1, entity2, …, entityK]

                Then analyze the text thoroughly so that you can find all the relationships between the entities. Find relationships only in the above list. There can be multiple relationships between two entities.
                Add relationships that anyone can agree with your decision. Bring the entity name directly from the original text.

                Here are the examples:
                """,
            ),
            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            MessagesPlaceholder("examples"),  # <-- EXAMPLES!
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
            ("human", "input: {input}"),
        ]
    )

    runnable = prompt | llm.with_structured_output(
        schema=Data,
        method="function_calling",
        include_raw=False,
    )

    with get_openai_callback() as cb:
        res = runnable.batch([{"input": text, "examples": messages} for text in texts], config={"max_concurrency": 3})
        print(cb)
    return res


def preprocess_text(file_path):
    """Remove Chinese characters and unnecessary special characters from the text."""
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        # Removing specific pattern (e.g., phone numbers) and non-Korean, non-English characters
        text = re.sub(
            r"\d{2,}", "", text
        )  # example to remove digits appearing in length of 2 or more
        text = re.sub('[^a-zA-Z0-9ㄱ-ㅣ가-힣., ·"]', "", text)
    return text


def process_file(file_path, res):
    print(f"Starting to process file: {file_path}")
    llm = ChatOpenAI(
        # Consider benchmarking with a good model to get
        # a sense of the best possible quality.
        model="gpt-4o",
        # Remember to set the temperature to 0 for extractions!
        temperature=0,
        verbose=True,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    input_batches = []
    num_sentences_per_batch = 5
    """Process a single file."""
    print(f"Processing {file_path}")
    text = preprocess_text(file_path)
    print(f"Preprocessed text: {text[:100]}...")
    sentences = text.split(".")
    print(sentences)
    for i in range(0, len(sentences), num_sentences_per_batch):
        input_batches.append(
            ".".join(sentences[i : min(i + num_sentences_per_batch, len(sentences))])
        )
    relationships = get_relationships_from_text(llm, input_batches)
    res = export_data_list(relationships, res, input_batches)
    print(f"Output added to res")


def main(directory):
    """Process all text files in the directory."""
    file_name_list = []
    files_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                file_name_list.append(file.split(".")[0])
                files_list.append(os.path.join(root, file))

    print(f"Total .txt files found: {len(files_list)}")

    res = {"data": []}

    # # Set up multiprocessing
    # with Pool(processes=os.cpu_count()) as pool:
    #     pool.starmap(process_file, [(file, res) for file in files_list])
    
    for i in range(len(files_list)):
        process_file(files_list[i], res)
        output_file = file_name_list[i] + ".json"
        export_data_list_as_json(res, output_file)



if __name__ == "__main__":
    try:
        main('data')
    except Exception as e:
        print(f"An error occurred: {e}")
