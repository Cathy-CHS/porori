from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import uuid
from typing import Dict, List, TypedDict, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

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
    head_person: str = Field(
        default=None, description="The person in the head of the relationship."
    )
    tail_person: str = Field(
        default=None, description="The person in the tail of the relationship."
    )
    relationship: str = Field(
        default=None, description="The relationship between the two people."
    )


class Entities(BaseModel):
    """list of people."""

    entities: List[str] = Field(default=None, description="The list of the people.")


class Data(BaseModel):
    """Extracted data about people."""

    # Creates a model so that we can extract multiple entities.
    # data: Dict["entities_list":Entities, "relationship_list":Relationship]
    entities_list: List[str]
    relationship_list: List[Relationship]


# Define Examples


class Example(TypedDict):
    """A representation of an example consisting of text input and expected tool calls.

    For extraction, the tool calls are represented as instances of pydantic model.
    """

    input: str  # This is the example text
    tool_calls: List[BaseModel]  # Instances of pydantic model that should be extracted


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

    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    openai_tool_calls = []
    for tool_call in example["tool_calls"]:
        openai_tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    # The name of the function right now corresponds
                    # to the name of the pydantic model
                    # This is implicit in the API right now,
                    # and will be improved over time.
                    "name": tool_call.__class__.__name__,
                    "arguments": tool_call.json(),
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


examples = [
    (
        """
     태조가 수창궁에서 왕위에 올랐다. 이보다 먼저 이달 12일에 공양왕이 장차 태조의 사제로 거둥하여 술자리를 베풀고 태조와 더불어 동맹하려고 하여 의장이 이미 늘어섰는데, 시중 배극렴 등이 왕대비에게 아뢰었다."지금 왕이 혼암하여 임금의 도리를 이미 잃고 인심도 이미 떠나갔으므로, 사직과 백성의 주재자가 될 수 없으니 이를 폐하기를 청합니다."마침내 왕대비의 교지를 받들어 공양왕을 폐하기로 일이 이미 결정되었는데, 남은이 드디어 문하 평리 정희계와 함께 교지를 가지고 북천동의 시좌궁 에 이르러 교지를 선포하니, 공양왕이 부복하고 명령을 듣고 말하기를,"내가 본디 임금이 되고 싶지 않았는데 여러 신하들이 나를 강제로 왕으로 세웠습니다. 내가 성품이 불민하여 사기를 알지 못하니 어찌 신하의 심정을 거스린 일이 없겠습니까"하면서, 이내 울어 눈물이 두서너 줄기 흘러내리었다. 마침내 왕위를 물려주고 원주로 가니, 백관이 국새를 받들어 왕대비전에 두고 모든 정무를 나아가 품명하여 재결하였다. 13일임진에 대비가 교지를 선포하여 태조를 감록국사로 삼았다.
     """,
        Data(
            entities_list=["태조", "공양왕", "배극렴", "왕대비", "남은", "정희계"],
            relationship_list=[
                Relationship(
                    head_person="태조", tail_person="공양왕", relationship="P1365"
                ),
                Relationship(
                    head_person="태조", tail_person="공양왕", relationship="P155"
                ),
                Relationship(
                    head_person="공양왕", tail_person="태조", relationship="P1366"
                ),
                Relationship(
                    head_person="공양왕", tail_person="태조", relationship="P156"
                ),
            ],
        ),
    ),
    (
        """
     이천우를 붙잡고 겨우 침문 밖으로 나오니 백관이 늘어서서 절하고 북을 치면서 만세를 불렀다. 태조가 매우 두려워하면서 스스로 용납할 곳이 없는 듯하니, 극렴 등이 합사하여 왕위에 오르기를 권고하였다."나라에 임금이 있는 것은 위로는 사직을 받들고 아래로는 백성을 편안하게 할 뿐입니다. “
     """,
        Data(
            entities_list=["이천우", "태조", "극렴"],
            relationship_list=[
                Relationship(
                    head_person="극렴", tail_person="태조", relationship="P737"
                ),
            ],
        ),
    ),
    (
        """
     선공감에 명하여 양청의 보첨을 짓게 하니, 선공감에서 아뢰었다."옛날에는 조은도가 선공감에 소속되었으므로, 매년 가을에 갈대를 베어서 국가의 용도에 공급하였사오나, 지금은 급전사에서 과전에 이를 소속시켜 참찬문하부사 정희계에게 주었으니, 양청의 차양은 갈대로써 덮기는 어렵겠습니다."임금이 말하기를,"고려 왕조의 공양군이 이 토지를 사사로이 그 아들에게 주었으니 좋은 일이 아닌데, 지금 급전사에서 그대로 과전에 소속시켜서 국가의 용도는 돌보지도 않고 재상에게 아첨을 구하니 옳은 일이 아니다."하고는, 즉시 급전사 장무 이재를 순군옥에 내려 가두고, 선공감 승 박자량으로 하여금 답험하게 하였다.
     """,
        Data(
            entities_list=["조은도", "정희계", "공양군", "이재", "박자량"],
            relationship_list=[
                Relationship(
                    head_person="조은도", tail_person="선공감", relationship="P361"
                ),
            ],
        ),
    ),
    (
        """
     간관이 상소하였는데, 대략은 이러하였다."그윽이 보건대, 상의중추원사 유양이 전에 계림에 있을 때에, 왜구가 투항할 즈음을 당하여 단기로 가서 적을 보고 화복으로 달래어, 왜노로 하여금 자식을 볼모로 바치고 정성을 다하게 하였으니, 마땅히 더욱 부지런히 하고 게을리 하지 말아서 그 공을 이루어야할 터인데, 왜선이 와서 정박한 뒤에 병이 심하다는 핑계로 이해의 기미를 조금도 생각하지 않고, 중 의운을 보내어 의심이 나서 도망하여 돌아가게 하였고, 또 중이 왕래한 형적을 비밀히 하여 조정에 나타내어 고하려 하지 않고, 그 까닭을 핵문하여도 말을 놀리는 것이 교묘하고 거짓되어 사정을 나타내지 않으니, 자못 간사하고 속이는 데에 가깝습니다. 원컨대 유사로 하여금 직첩을 거두고 정상과 사유를 밝게 국문 하소서."임금이 단지 외방에 부처하고, 의운의 출현을 기다려 빙문하여 사실을 조사하게 하였다.
     """,
        Data(
            entities_list=["간관", "유양", "왜구", "의운"],
            relationship_list=[
                Relationship(
                    head_person="유양", tail_person="왜구", relationship="P710"
                ),
                Relationship(
                    head_person="의운", tail_person="유양", relationship="P156"
                ),
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
{
"P17": "다음 국가의 것임 (country)",
"P131": "다음 행정구역에 위치함 (located in)",
"P530": "수교국 (diplomatic relation)",
"P150": "하위 행정구역",
"P47": "경계를 접하고 있음 (shares border with)",
"P106": "직업 (occupation)",
"P27": "국적 (country of citizenship)",
"P461": "반대 개념 (opposite of)",
"P279": "하위 개념 (subclass of)",
"P495": "해당 개체가 처음 유래되거나 만들어진 국가 (country of origin)",
"P641": "관련 스포츠 종목 (related sports)",
"P156": "시간적 또는 공간적으로 뒤에 오는 것 (followed by)",
"P155": "시간적 또는 공간적으로 앞에 오는 것 (follows)",
"P527": "해당 개체가 다음으로 이루어져 있음 (has part)",
"P361": "해당 개체가 다음의 일부분임 (part of)",
"P1376": "해당 지역이 다음의 수도임 (capital of)",
"P36": "해당 국가의 수도가 다음 지역임 (capital)",
"P118": "다음 리그에 출전함 (league)",
"P1889": "다음과 다르지만 같은 의미인 것처럼 혼동되는 항목 (different from)",
"P31": "다음 종류에 속함 (instance of)",
"P175": "연주자·가수 (performer)",
"P463": "다음 공동체의 구성원임 (member of)",
"P54": "해당 개체의 현재 또는 과거의 소속 팀 (member of sports team)",
"P138": "해당 개체의 명칭은 다음에서 유래되었음 (named after)",
"P81": "해당 정거장을 거쳐가는 노선 (connecting line)",
"P40": "해당 개체의 자녀 (child)",
"P159": "해당 기관·단체의 본부·본사가 있거나 있었던 곳 (headquarters location)",
"P136": "해당 개체의 장르 (genre)",
"P171": "해당 개체가 다음의 하위 분류군임 (parent taxon)",
"P22": "해당 개체의 친아버지 (father)",
"P26": "해당 개체의 배우자 (spouse)",
"P3373": "해당 개체의 친형제자매 (sibling)",
"P50": "해당 개체의 저자 (author)",
"P30": "해당 개체가 속한 대륙 (continent)",
"P1532": "해당 인물이 대표하는 국가 (country for sport)",
"P178": "해당 개체를 개발한 주체 (developer)",
"P413": "해당 선수가 맡은 포지션 (position played on team / speciality)",
"P800": "대표 작품 (notable work)",
"P1365": "해당 개체는 다음으로부터 이어짐 (replaces)",
"P276": "해당 개체의 위치 (location)",
"P1366": "해당 개체는 다음으로 이어짐 (replaced by)",
"P19": "태어난 곳 (place of birth)",
"P449": "해당 개체가 다음에서 본방송을 함 (original broadcaster)",
"P710": "해당 사건에 다음이 참여함 (participant)",
"P2936": "해당 장소나 사건에서 다음 언어가 주로 사용됨 (language used)",
"P1001": "해당 개체가 영향력을 미치는 관할 구역 (applies to jurisdiction)",
"P140": "해당 개체와 연관된 종교 (religion)",
"P206": "해당 지역이 뻗고 있는 유역 (located in or next to body of water)",
"P1056": "해당 개체의 제품 (product or material produced)",
"P20": "사망한 곳 (place of death)",
"P6": "해당 정부 또는 지자체의 수장 (head of government)",
"P123": "해당 개체의 발행 주체 (publisher)",
"P1830": "해당 개체가 소유한 것 (owner of)",
"P127": "해당 개체의 소유자 (owned by)",
"P1659": "해당 개체와 같이 등장하는 개념 (see also)",
"P112": "해당 개체의 설립자 (founded by)",
"P101": "전문 분야 (field of work)",
"P3095": "다음이 이것을 실천함 (practiced by)",
"P749": "해당 개체가 다음의 산하 기관임 (parent organization)",
"P1696": "반대 속성 (inverse property)",
"P137": "해당 개체의 운영 주체 (operator)",
"P2789": "해당 개체가 다음과 연결되어 있음 (connects with)",
"P706": "해당 개체가 위치한 지형 (located in/on physical feature)",
"P3842": "현재 다음 행정구역에 위치함 (located in present-day administrative territorial entity)",
"P39": "해당 인물의 현재 또는 과거의 직위 (position held)",
"P425": "직업의 분야 (field of this occupation)",
"P1336": "해당 지역에 대해 영유권 또는 관할권을 주장하는 국가·지역·단체 (territory claimed by)",
"P108": "해당 인물을 고용한 곳 (employer)",
"P172": "다음은 해당 개체의 민족에 속함 (ethnic group)",
"P737": "다음에게서 영향을 받음 (influenced by)",
"P176": "해당 개체의 제조사 (manufacturer)",
"P102": "해당 인물의 소속 정당 (member of political party)",
"P35": "해당 국가의 원수 (head of state)",
"P3730": "해당 개체의 바로 윗 계급 (next higher rank)",
"P1687": "위키데이터 속성 (Wikidata property)",
"P945": "해당 개체가 섬기는 국가나 세력 (allegiance)",
"P264": "해당 개체의 음반 레이블 (record label)",
"P161": "해당 작품에 출연한 배우 (cast member)",
"P69": "해당 인물의 학교 (educated at)",
"P190": "자매 결연 도시 (twinned administrative body)",
"P355": "해당 개체의 산하 기관 (subsidiary)",
"P2341": "해당 개체가 다음 대상의 고유의 것임 (indigenous to)",
"P664": "주최자 (organizer)",
"P407": "해당 저작물이나 명칭에서 사용한 언어 (language of work or name)",
"P793": "해당 개체에게 있었던 주목할 만한 사건 (significant event)",
"P840": "공간적 배경 (narrative location)",
"P1441": "해당 개체가 다음 작품에 등장함 (present in work)",
"P607": "해당 개체가 참전한 전쟁 (conflict)",
"P197": "해당 역과 인접한 역 (adjacent station)",
"P205": "해당 유역을 뻗고 있는 국가 (basin country)",
"P162": "해당 작품의 제작자 (producer)",
"P807": "해당 개체가 다음으로부터 갈라져 나와 시작된 것임 (separated from)",
"P1269": "해당 주제의 상위 주제 (facet of)",
"P170": "해당 개체의 창작자 (creator)",
"P97": "작위 (noble title)",
"P750": "해당 개체의 유통을 다음이 맡음 (distributed by)",
"P551": "해당 인물의 거주지 (residence)"
}

The output shape will be list of dictionaries like this.

First generate the list of entities in the input text. Only find person entities. list your findings in this format:

[entity1, entity2, …, entityK]

Then analyze the text thoroughly so that you can find all the relationships between the entities. Find relationships only in the above list. There can be multiple relationships between two entities.

Then write your result starting with '<results>'. e.g.
<results>
[
{
head: head_entity,
tail: tail_entity,
class: relationship_class_label
}
…
]

Here are the examples:
""",
        ),
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        MessagesPlaceholder("examples"),  # <-- EXAMPLES!
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        ("human", "{text}"),
    ]
)

# Testing the prompt
text = """
    사간원에서 김덕원의 일을 아뢰니, 그대로 따랐다. 우의정 민정중이 하루 전에 입시하여 말하다가, 김덕원의 일에 미치자 임금에게 빨리 대간이 아뢴 대로 따를 것을 권하고, 또 아뢰기를,"김덕원이 부지런하고 성실하여 직책을 잘 수행했다는 칭찬이 조금 있으니, 성상께서 대간의 아룀을 윤허하지 않으심은, 진실로 인재를 사랑하고 아끼는 뜻에서 나왔겠지만, 공의가 이미 발표된 뒤에는 또한 시비를 명백히 하여 악을 징계하고 선을 장려하는 터전을 삼지 않을 수 없습니다.
"""
# res = prompt.invoke({"text": text, "examples": messages})

# print(res)


# We will be using tool calling mode, which
# requires a tool calling capable model.
llm = ChatOpenAI(
    # Consider benchmarking with a good model to get
    # a sense of the best possible quality.
    model="gpt-4-turbo",
    # Remember to set the temperature to 0 for extractions!
    temperature=0,
    verbose=True,
    api_key=os.getenv("OPENAI_API_KEY"),
)

runnable = prompt | llm.with_structured_output(
    schema=Data,
    method="function_calling",
    include_raw=False,
)

for _ in range(5):
    print(runnable.invoke({"text": text, "examples": messages}))
