import re


# 파일에서 텍스트 읽기 함수
def read_file(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        return file.read()


# 파일에 텍스트 쓰기 함수
def write_file(file_name, data):
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(data)


# 파일 처리 함수
def process_text(file_name, output_file_name):
    text = read_file(file_name)  # 파일에서 텍스트 읽기
    processed_text = re.sub(r"0\d{2}", "", text)
    # 정규 표현식으로 추가 처리, 필요 없는 문자 제거
    processed_text = re.sub('[^a-zA-Z0-9ㄱ-ㅣ가-힣., ·"]', "", processed_text)
    write_file(output_file_name, processed_text)  # 결과를 새 파일에 저장


# 실행 코드
input_file_name = "src/input.txt"  # 입력 파일 이름 지정
output_file_name = "src/output.txt"  # 출력 파일 이름 설정

process_text(input_file_name, output_file_name)  # 파일 처리 실행
print(
    "처리가 완료되었습니다. 결과는 {} 파일에 저장되었습니다.".format(output_file_name)
)
