# app/controllers/fetch_patents.py

import requests
import re
from bs4 import BeautifulSoup

def fetch_patent_metadata(patent_id: str) -> dict:
    """
    Fetch key metadata and full description of a patent from Google Patents.

    Args:
        patent_id (str): Patent number (e.g., "US1234567B2")

    Returns:
        dict: {
            'patent_id': str,
            'title': str,
            'assignee': str,
            'description': str
        }
    """
    url = f"https://patents.google.com/patent/{patent_id}/en?oq={patent_id}"

    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=10)

        # ❗ 404 or 기타 비정상 응답이면 None 반환
        if response.status_code != 200:
            print(f"[fetch_patent_metadata] Patent not found: {patent_id} (status code: {response.status_code})")
            return None

    except Exception as e:
        print(f"[fetch_patent_metadata] Request failed for {patent_id}: {e}")
        return None  # 특허 없는 경우 None 반환

    soup = BeautifulSoup(response.content, "html.parser")

    # Title
    title = soup.find("span", attrs={"itemprop": "title"})
    title_text = title.get_text(strip=True) if title else "(no title found)"

    # Assignee
    assignee = soup.find("dd", attrs={"itemprop": "assigneeOriginal"})
    assignee_text = assignee.get_text(strip=True) if assignee else "(no assignee found)"

    # Description
    description_div = soup.find("section", {"itemprop" :"description"})
    description_text = description_div.get_text(separator=".", strip=True) if description_div else "(no description found)"

    # Clean up the description text
    description_text = re.sub(r"[^a-zA-Z0-9\s.,:;!?()\-\"'%/[\]]", "", description_text)  # Remove extra whitespace

    return {
        "patent_id": patent_id,
        "title": title_text,
        "assignee": assignee_text,
        "description": description_text
    }
