from playwright.sync_api import sync_playwright
import logging
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel("INFO")
# logger timestamp
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

item_page_headers = ["개설", "가계", "활동 사항", "묘소", "참고문헌", "관계망"]


def extract_items_with_playwright(url: str):
    """
    Extract items and urls from the given URL


    Args:
        url (str): _description_

    Returns:
        _type_: _description_
    """
    with sync_playwright() as p:
        # Start a browser session
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        # Navigate to the initial URL
        res = page.goto(url)
        if res.status != 200:
            logger.error(f"Failed to load the page: {url}")
            return []
        logger.info(f"Loaded the page: {url}")
        items = []

        while True:
            # Wait for the list to ensure it's loaded
            page.wait_for_selector("ul")

            # Extract the data from the current page
            list_items = page.query_selector_all("ul > li")
            for li in list_items:
                link = li.query_selector("a")
                if link:
                    name = link.text_content()
                    item_url = link.get_attribute("href")
                    items.append((name, "https://dh.aks.ac.kr" + item_url))

            logger.info(f"Extracted {len(items)} items from the page: {page.url}")
            # Try to find the 'next page' button
            next_page = page.query_selector("a:has-text('다음 페이지')")
            if next_page:
                # Click the next page button and wait for the next page to load
                logger.info("Found the next page button and clicking it.")
                next_page.click()

                logger.info(f"Loaded the next page: {page.url}")
            else:
                # Exit the loop if no next page button is found
                logger.info("No next page button found. Exiting the loop.")
                break

        # Close the browser
        logger.info("Closing the browser.")
        browser.close()
        return items


def fetch_descriptions(
    urls,
):
    """
    Fetch paragraphs between headers from a given URL

    Args:
        url (str): The URL to fetch the data from

    Returns:
        dict: A dictionary containing the extracted paragraphs.

    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the URL
        res_items = []
        res_descriptions = []
        res_data = []
        for url in tqdm(urls, desc="Extracting data from URLs"):
            try:
                page.goto(url)
            except Exception as e:
                logger.error(f"Failed to load the page: {url}")
                logger.error(e)
                res_items.append({})
                res_descriptions.append("")
                res_data.append([])
                continue

            items = dict()
            h2_elements = page.query_selector_all("h2")
            h2_texts = [element.inner_text() for element in h2_elements]
            description = ""
            # Extract paragraphs between headers
            for i in range(len(h2_texts) - 1):

                start_header_text = h2_texts[i]
                end_header_text = h2_texts[i + 1]

                # JavaScript to extract all <p> texts between two <h2> elements by their text content
                paragraphs = page.evaluate(
                    f"""() => {{
                    const headers = Array.from(document.querySelectorAll('h2'));
                    const startHeaderIndex = headers.findIndex(header => header.textContent.includes('{start_header_text}'));
                    const endHeaderIndex = headers.findIndex((header, index) => header.textContent.includes('{end_header_text}') && index > startHeaderIndex);

                    const elements = [];
                    if (startHeaderIndex !== -1 && endHeaderIndex !== -1) {{
                        let element = headers[startHeaderIndex].nextElementSibling;
                        while (element && element !== headers[endHeaderIndex]) {{
                            if (element.tagName === 'P') {{
                                elements.push(element.textContent);
                            }}
                            element = element.nextElementSibling;
                        }}
                    }}
                    return elements;
                }}"""
                )

                paragraphs = "\n".join(paragraphs)
                description += "##" + start_header_text + "##" + ": " + paragraphs
                items[start_header_text] = paragraphs

            # 기사 연계 정보 추출
            data = page.evaluate(
                """() => {
                    
                    const rows = Array.from(document.querySelectorAll('tr'));
                    const startIndex = rows.findIndex(row => row.querySelector('th') && row.querySelector('th').textContent.includes('조선왕조실록 기사 연계'));

                    const elements = [];
                    if (startIndex !== -1) {
                        
                        // Collect all subsequent rows until another <th colspan="2"> is found or end of table
                        for (let i = startIndex + 1; i < rows.length; i++) {
                            const links = rows[i].querySelectorAll('a');
                            if (links.length > 0) {
                                Array.from(links).forEach(link => {
                                    elements.push({text: link.textContent, href: link.href});
                                });
                            }
                        }
                    }
                    return elements;
                }"""
            )
            res_items.append(items)
            res_descriptions.append(description)
            res_data.append(data)
        browser.close()
    return res_items, res_descriptions, res_data


def serialize_as_table(item_urls):
    """
    Serialize the extracted data as a table

    Args:
        item_urls (_type_): _description_

    Returns:
        _type_: _description_
    """
    names = []
    wiki_urls = []
    wiki_descriptions = []
    siloc_urls = []

    _, descriptions, siloc_links = fetch_descriptions(item_urls[:, 1])

    for i, description in enumerate(descriptions):
        name = item_urls[i][0]
        item_url = item_urls[i][1]

        siloc_links_per_item = siloc_links[i]
        if siloc_links_per_item:
            for siloc_link in siloc_links_per_item:
                names.append(name)
                wiki_urls.append(item_url)
                wiki_descriptions.append(description)
                siloc_urls.append(siloc_link["href"])
        else:
            names.append(name)
            wiki_urls.append(item_url)
            wiki_descriptions.append(description)
            siloc_urls.append("")

    df = pd.DataFrame(
        {
            "name": names,
            "wiki_url": wiki_urls,
            "wiki_description": wiki_descriptions,
            "siloc_url": siloc_urls,
        }
    )
    return df


if __name__ == "__main__":
    # Example usage:
    url = "https://dh.aks.ac.kr/sillokwiki/index.php?title=%EB%B6%84%EB%A5%98:%EC%9D%B8%EB%AC%BC"
    items = extract_items_with_playwright(url)
    items_df = pd.DataFrame(items, columns=["name", "url"])
    items_df.to_csv("실록위키_인물_이름url.csv", index=False, encoding="utf-8")
    items_df = pd.read_csv("실록위키_인물_이름url.csv")
    items = items_df.values
    df = serialize_as_table(items)
    df.to_csv("실록위키_실록url까지.csv", index=False)
