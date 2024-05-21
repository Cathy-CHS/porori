import os
from urllib.request import urlopen
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# 메인 페이지 열기
main_url = "https://sillok.history.go.kr/main/main.do"

# 메인 페이지 열기

def king_links(main_url) -> list[(str, str)]:
    response = urlopen(main_url)
    soup = BeautifulSoup(response, "html.parser")
    # 'ul' 태그 내의 'li' 태그에서 모든 'a' 태그 링크 추출
    classes = ["m_cont_top01", "m_cont_top02", "m_cont_top03", "m_cont_top04"]
    ul_tags = soup.find_all("ul", {"class": classes})

    # 왕들의 링크 수집
    king_links = []
    for ul in ul_tags:
        li_tags = ul.find_all("li")
        for li in li_tags:
            a_tag = li.find("a")
            if a_tag and 'href' in a_tag.attrs:
                match = re.search(r"javascript:search\('([^']+)'\)", a_tag['href'])
                if match:
                    page_id = match.group(1)
                    king_name_full = a_tag.get_text().strip()
                    # 괄호 및 그 안의 내용 제거
                    king_name = re.sub(r"\s*\(.*?\)", "", king_name_full).strip()
                    # "1대 태조" 형식을 그대로 남기기
                    king_name = re.sub(r"(\d+[대]\s*.*)", r"\1", king_name)
                    page_link = f"https://sillok.history.go.kr/search/inspectionMonthList.do?id={page_id}"
                    king_links.append((king_name, page_link))

    return king_links

# 연도별 기록 링크 추출 함수
def extract_month_links(king_page_url) -> list[str]:
    response = urlopen(king_page_url)
    soup = BeautifulSoup(response, "html.parser")

    # 연도별 기록이 있는 'ul' 태그들 찾기
    ul_tags = soup.find_all("ul", class_=re.compile(r"king_year2"))

    # 연도별 링크 수집
    month_links = []
    for ul in ul_tags:
        li_tags = ul.find_all("li")
        for li in li_tags:
            a_tag = li.find("a")
            if a_tag and 'href' in a_tag.attrs:
                match = re.search(r"javascript:search\('([^']+)'\s*,\s*'([^']+)'\s*,\s*'([^']+)'\)", a_tag['href'])
                if match:
                    king_id = match.group(1)
                    cnt = match.group(2)
                    level = match.group(3)
                    # 링크 생성
                    month_link = f"https://sillok.history.go.kr/search/inspectionDayList.do?id={king_id}&level={level}"
                    month_links.append(month_link)

    return month_links

def extract_text_links(month_page_url) -> list[(str, str)]:
    response = urlopen(month_page_url)
    soup = BeautifulSoup(response, "html.parser")

    # 클래스가 "ins_list_main"인 <dl> 태그 찾기
    dl_tag = soup.find("dl", {"class": "ins_list_main"})
    
    # 해당 <dl> 태그 내에서 <dt> 태그의 텍스트 추출
    dt_tag = dl_tag.find("dt") if dl_tag else None
    year_month_full = dt_tag.get_text(strip=True) if dt_tag else ""
    match = re.search(r"[가-힣]+\s*\d+년\s*윤?\s*\d+월|즉위년\s*\d+월", year_month_full)
    year_month = match.group(0) if match else ""

    # 해당 <dl> 태그 내에서 <li> 태그 내의 모든 <a> 태그 찾기
    a_tags = dl_tag.find_all("a") if dl_tag else []

    # 각 <a> 태그의 href 속성 추출 및 출력
    base_url = "https://sillok.history.go.kr"
    links = []

    for a in a_tags:
        if 'href' in a.attrs:
            link = a['href']
            match = re.search(r"javascript:searchView\('([^']+)'\)", link)
            if match:
                id = match.group(1)
                full_url = f"{base_url}/id/{id}"
                links.append((year_month, full_url))

    return links

def extract_records_from_month_page(link):
    response = urlopen(link)
    soup = BeautifulSoup(response, "html.parser")
    
    # 클래스가 "ins_view_in ins_left_in"인 <div> 태그 찾기
    div_tag = soup.find("div", {"class": "ins_view_in ins_left_in"})

    # 해당 <div> 태그 내에서 클래스가 "paragraph"인 모든 <p> 태그 찾기
    paragraphs = div_tag.find_all("p", {"class": "paragraph"}) if div_tag else []

    # 하나의 글로 합치기 위해 빈 문자열 변수 선언
    full_text = ''

    # 각 <p> 태그의 텍스트 추출 및 결합
    for paragraph in paragraphs:
        text = ''
        for element in paragraph.children:
            if isinstance(element, NavigableString):
                text += element
            elif isinstance(element, Tag) and element.name == 'span':
                text += element.get_text()
        full_text += text.strip() + ' '

    return full_text.strip()


def save_record_to_file(root_directory, king_name, term, record_number, text):
    directory = os.path.join(root_directory, king_name, term)
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{record_number}.txt")
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

# 병렬 처리 예제
def main():
    # 이미 완료된 왕 목록
    completed_kings_list = [
        # "1대 태조", 
        # "2대 정종", 
        # "3대 태종", 
        # "4대 세종",
        # "5대 문종",
        # "6대 단종",
        # "7대 세조",
        # "8대 예종",
        # "9대 성종",
        # "10대 연산군",
        # "11대 중종",
        # "12대 인종",
        # "15대 광해군중초본",
        # "17대 효종"
        ]  
    king_list = king_links(main_url)
    root_directory = "records"
    log_file = open("log.txt", "w", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_king = {executor.submit(extract_month_links, king[1]): king for king in king_list if king[0] not in completed_kings_list}

        for future in as_completed(future_to_king):
            king_name, king_link = future_to_king[future]
            try:
                month_links = future.result()
                future_to_month = {executor.submit(extract_text_links, month): (king_name, month) for month in month_links}

                for future in as_completed(future_to_month):
                    king_name, month = future_to_month[future]
                    try:
                        text_links = future.result()
                        future_to_record = {executor.submit(extract_records_from_month_page, link): (king_name, term, idx + 1) for idx, (term, link) in enumerate(text_links)}

                        for future in as_completed(future_to_record):
                            king_name, term, record_number = future_to_record[future]
                            try:
                                record_text = future.result()
                                log_file.write(f"Saving record {record_number} for {king_name} {term}\n")
                                print(f"Saving record {record_number} for {king_name} {term}")
                                save_record_to_file(root_directory, king_name, term, record_number, record_text)
                            except Exception as e:
                                log_file.write(f"Error processing record {record_number} for {king_name} {term}: {e}\n")
                                print(f"Error processing record {record_number} for {king_name} {term}: {e}")

                    except Exception as e:
                        log_file.write(f"Error processing month {month} for {king_name}: {e}\n")
                        print(f"Error processing month {month} for {king_name}: {e}")

            except Exception as e:
                log_file.write(f"Error processing king {king_name}: {e}\n")
                print(f"Error processing king {king_name}: {e}")
    log_file.close()

if __name__ == "__main__":
    main()