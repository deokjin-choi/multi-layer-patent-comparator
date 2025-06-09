# path_helper.py
import os

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_NAME = "patent_compare"
BASE_DIR = CURRENT_FILE[:CURRENT_FILE.index(PROJECT_NAME) + len(PROJECT_NAME)]

# 데이터 디렉토리 설정
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
