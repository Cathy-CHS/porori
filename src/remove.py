import re

class NeoBuri:
    def __init__(self, file):
        self.input_file = file
        self.output_file = 'output.txt'

    # # 파일에서 텍스트 읽기 함수
    # def read_file(self):
    #     with open(self.file, 'r', encoding='utf-8') as f:
    #         return f.read()

    # # 파일에 텍스트 쓰기 함수
    # def write_file(self, data):
    #     with open(self.file, 'w', encoding='utf-8') as file:
    #         file.write(data)

    # 파일 처리 함수
    def process_text(self):
        input = open(self.input_file, 'r', encoding='utf-8')
        text = input.read()
        # text = self.read_file(self.file)  # 파일에서 텍스트 읽기
        processed_text = re.sub(r'0\d{2}', '', text) # 주석 제거
        processed_text = re.sub('[^a-zA-Z0-9ㄱ-ㅣ가-힣., ·"]', '', processed_text)  # 한자 및 불필요 문자 제거
        # output = open(self.output_file, 'w', encoding='utf-8')
        # output.write(processed_text)
        # print("처리가 완료되었습니다. 결과는 {} 파일에 저장되었습니다.".format(self.output_file))
        # input.close()
        # write_file(output_file_name, processed_text)  # 결과를 새 파일에 저장

        return processed_text

# # 실행 코드
# input_file_name = 'input.txt'  # 입력 파일 이름 지정
# output_file_name = 'output.txt'  # 출력 파일 이름 설정

# process_text(input_file_name, output_file_name)  # 파일 처리 실행
