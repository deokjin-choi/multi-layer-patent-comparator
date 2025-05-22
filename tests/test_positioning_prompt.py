# tests/test_positioning_prompt.py
import sys
import os
import importlib


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.controllers.positioning import generate_positioning_summary

def test_generate_positioning_summary():
    # 가상의 특허번호
    patent_id = "TEST1234567"

    # 가상의 요약 정보 (LLM에 넣을 입력)
    summary = {
        "problem": "Deep neural networks suffer from inefficient matrix multiplication operations.",
        "solution_function": "Introduces burst-mode memory access and dual multiplying units for parallel computation.",
        "solution_structure": "System includes two multiplying units, inner buffer circuits, and an outer buffer for final results.",
        "solution_implementation": "Data is loaded into buffer circuits, multiplied in parallel, and results summed via shared adders.",
        "effect": "Improves inference speed and reduces energy consumption in deep learning accelerators."
    }

    # 테스트용 함수 호출
    result = generate_positioning_summary(patent_id, summary)

    # 결과 출력
    print(f"\n🧪 Positioning Summary for {patent_id}:")
    for k, v in result.items():
        print(f"{k}: {v}")

    # 간단한 결과 검증 (터미널 확인용)
    assert "functional_purpose" in result
    assert "technical_uniqueness" in result
    assert "strategic_value" in result

if __name__ == "__main__":
    test_generate_positioning_summary()
