import os
import re
from remove import NeoBuri

def get_all_txt_files(root_dir):
    txt_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                txt_files.append(os.path.join(dirpath, filename))
    return txt_files

def extract_date_info(file_path):
    # 파일 경로에서 연도와 월 정보를 추출
    match = re.search(r'((즉위년|\d+년)\s*(윤?\d+월))', file_path)
    if match:
        year_month = match.group(1)
        year_str = match.group(2)
        month_match = re.search(r'(윤?\d+)월', year_month)
        month_str = month_match.group(1)
        
        # 윤달 처리
        is_leap = '윤' in month_str
        month = int(month_str.replace('윤', ''))

        # '즉위년'을 -1로 설정하여 연도 정렬 시 가장 앞에 오도록 함
        if '즉위년' in year_str:
            year = -1
        else:
            year = int(re.search(r'(\d+)년', year_str).group(1))
        
        return year, month, is_leap
    return None

def sort_files(files):
    # 연도와 월 정보를 기준으로 파일 그룹화 및 정렬
    grouped_files = {}
    for file in files:
        date_info = extract_date_info(file)
        if date_info:
            year, month, is_leap = date_info
            key = (year, month, is_leap)
            if key not in grouped_files:
                grouped_files[key] = []
            grouped_files[key].append(file)
    
    # 각 그룹 내에서 파일 이름을 숫자 순서로 정렬
    sorted_groups = []
    for key in sorted(grouped_files.keys()):
        sorted_files = sorted(grouped_files[key], key=lambda f: int(re.search(r'(\d+)', os.path.basename(f)).group()))
        sorted_groups.extend(sorted_files)
    
    return sorted_groups

def combine_txt_files(input_dir, output_file):
    # 모든 txt 파일 목록 가져오기
    files = get_all_txt_files(input_dir)

    # 연도와 월, 파일 이름 순으로 정렬
    sorted_files = sort_files(files)
    print(file for file in sorted_files)

    all_processed_text = []

    for input_file in sorted_files:
        neoburi = NeoBuri(input_file)
        processed_text = neoburi.process_text()
        all_processed_text.append(processed_text)
        print(f"Processed {input_file}")

    # 모든 텍스트를 하나의 문자열로 결합
    combined_text = ' '.join(all_processed_text)

    # 결합된 텍스트를 파일에 저장
    directory = os.path.dirname(output_file)
    os.makedirs(directory, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(combined_text)

kings = {
    1: "1대 태조",
    2: "2대 정종",
    3: "3대 태종",
    4: "4대 세종",
    5: "5대 문종",
    6: "6대 단종",
    7: "7대 세조",
    8: "8대 예종",
    9: "9대 성종",
    10: "10대 연산군",
    11: "11대 중종",
    12: "12대 인종",
    13: "13대 명종",
    14: "14대 선조",
    15: "15대 광해군중초본",
    16: "16대 인조",
    17: "17대 효종",
    18: "18대 현종",
    19: "19대 숙종",
    20: "20대 경종",
    21: "21대 영조",
    22: "22대 정조",
    23: "23대 순조",
    24: "24대 헌종",
    25: "25대 철종",
    26: "26대 고종",
    27: "27대 순종",
    28: "순종부록",

}

for i in range(1, 29):
    king_name = kings[i]
    input_dir = f'records/{king_name}'
    output_file = f'outputs/{king_name}.txt'
    combine_txt_files(input_dir, output_file)
