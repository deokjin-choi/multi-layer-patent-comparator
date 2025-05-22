# tests/patent_fetch.py

import os
import sys
import importlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 📌 reload 대상 모듈 import
import app.controllers.fetch_patents as fetch_patents

# 📌 수정 사항 즉시 반영
importlib.reload(fetch_patents)

def test_fetch():
    response = fetch_patents.fetch_patent_metadata("US11475102B2")
    print(response)

if __name__ == "__main__":
    test_fetch()
