import os
import json

RAW_DIR = "data/raw"
SUMMARY_DIR = "data/summary"

def load_cached_patent(patent_id: str) -> dict | None:
    file_path = os.path.join(RAW_DIR, f"{patent_id}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_patent_cache(patent_id: str, data: dict):
    os.makedirs(RAW_DIR, exist_ok=True)
    file_path = os.path.join(RAW_DIR, f"{patent_id}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# --------------------------
# 요약 결과 캐싱 함수
# --------------------------

def load_summary(patent_id: str) -> dict | None:
    """저장된 요약 결과를 불러옵니다."""
    summary_path = os.path.join(SUMMARY_DIR, f"{patent_id}.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_summary(patent_id: str, summary_data: dict):
    """요약 결과를 저장합니다."""
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    summary_path = os.path.join(SUMMARY_DIR, f"{patent_id}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
