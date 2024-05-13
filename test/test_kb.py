from dotenv import load_dotenv
import os, sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from knowledgebase.knowledgebase import EncyKoreaAPIEntity

load_dotenv()

DEFINITION = "한글 자음에서 첫 번째로 등장하는 글자. 기역."
DESCRIPTION = """
# 내용\r\n국어의 자음 가운데, 목젖으로 콧길을 막고 혀뿌리［舌根］를 높여 연구개(軟口蓋)를 막았다가 뗄 때 나는 소리, 즉 연구개 [파열음](/Article/E0059582)을 표기하는 데 쓰인다.\r\n\r\n그래서 해례본 제자해(制字解)에서는 혀뿌리가 목구멍을 막은 모양을 본떠서(象舌根閉喉之形) 이 글자를 만들었다고 설명하고, 이 소리를 아음(牙音)의 전청음(全淸音)에 소속시켜, 이 소리가 무성 · 무기음(無聲 · 無氣音)인 것을 보였다.\r\n\r\n그러나 이 소리는 어두(語頭)에서는 무성음으로 나고, 어중(語中)의 유성음 사이에서는 유성음으로 나며, 음\r\n절의 말음 자리에서는 연구개를 막은 혀뿌리를 떼지 않은 상태로 그친다.\r\n\r\nㄱ음에 대하여 『훈민정음』 본문 예의편(例義篇)에 의하면 “ㄱ은 어금닛소리니 군(君)자의 처음 나는 소리와 같다(ㄱ 牙音 如君字初發聲).”라고 풀이하였고, 『훈민정음언해』에서는 “ㄱᄂᆞᆫ : 엄쏘 · 리 · 니 君군ㄷ字 · ᄍᆞᆼ · 처 · ᅀᅥᆷ · 펴 · 아 · 나ᄂᆞᆫ 소 · 리 · ᄀᆞᄐᆞ · 니라”라고 설명하였다.\r\n\r\n[『훈몽자회』](/Article/E0065801) 범례에 처음으로 자모의 이름이 보여서 초성종성통용팔자(初聲終聲通用八字)란에, ‘ㄱ 其役(기역)’이라고 적혀 있다. 한글 자모의 첫 글자이면서 쉬운 글자이기 때문에, 우리나라 속담에서 아주 무식한 사람을 일컬을 때 ‘낫 놓고 기역자도 모른다.’고 하여, 낫의 모양이 기역(ㄱ)자처럼 생겼는데도 그렇게 쉬운 글자조차 모르는 사람을 비유하는 데 쓰고 있다.\r\n\r\nㄱ자는 훈민정음이 창제된 무렵에는 국어의 초성, 종성뿐만 아니라, ㆁ으로 끝나는 한자 아래에서 ‘兄ㄱᄠᅳ디’([용비어천가](/Article/E0039509) 8장)처럼 속격조사의 표기에도 쓰였다.
"""


def test_encykorea_api_entity():
    entity = EncyKoreaAPIEntity(
        "ㄱ", "E0000002", access_key=os.getenv("ENCYKOREA_API_KEY")
    )
    assert entity.name == "ㄱ"
    assert entity.entity_id == "E0000002"
    entity._get_content()

    assert entity.definition.strip() == DEFINITION.strip()
    assert entity.description.strip() == DESCRIPTION.strip()

    res = entity.get_context()
    assert res["name"] == "ㄱ"
    assert res["definition"].strip() == DEFINITION.strip()
    assert res["description"].strip() == DESCRIPTION.strip()


if __name__ == "__main__":
    test_encykorea_api_entity()
    print("All tests passed!")

    entity = EncyKoreaAPIEntity(
        "ㄱ", "E0000002", access_key=os.getenv("ENCYKOREA_API_KEY")
    )
    print(entity.get_context())
