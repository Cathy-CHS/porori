import json

cls_to_label_dict = {
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
    "P97": "작위 (noble title)",
}


def convert(json_path, converted_file_path, cls_to_label_dict):

    with open(json_path, "r", encoding="UTF-8-sig") as file:
        dictionary = json.load(file)
        for data in dictionary["data"]:
            for rel in data["relationships"]:
                try:
                    rel[2] = cls_to_label_dict[rel[2]]
                except KeyError:
                    print(f"{rel[2]} not found in cls_to_label_dict")
                    data["relationships"].remove(rel)

    with open(converted_file_path, "w") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)
    print("변환 완료")


if __name__ == "__main__":
    json_path = "src/rel_ext_data.json"
    converted_file_path = "src/rel_ext_data_converted.json"
    convert(json_path, converted_file_path, cls_to_label_dict)