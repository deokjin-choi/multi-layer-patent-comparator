# app/controllers/engine.py

from app.controllers.fetch_patents import fetch_patent_metadata

from app.utils.caching import (
    load_cached_patent,
    save_patent_cache,
    load_summary,
    save_summary
)

from app.controllers.summarize import summarize_patent
from app.controllers.positioning import analyze_positioning
from app.controllers.implementation import analyze_implementation_diff

# 미래 확장: positioning, implementation, strategy

def run_analysis(our_patent_id: str, competitor_patent_ids: list[str]) -> dict:

    def get_or_fetch_with_summary(patent_id):
        # 1. 캐시된 원문 불러오기 또는 새로 수집
        patent_data = load_cached_patent(patent_id)
        if not patent_data:
            patent_data = fetch_patent_metadata(patent_id)
            save_patent_cache(patent_id, patent_data)

        # 2. 요약 캐시 확인
        summary = load_summary(patent_id)
        if summary is None:
            summary = summarize_patent(patent_data["description"], patent_id, "v2")
            save_summary(patent_id, summary)

        patent_data["summary"] = summary
        return patent_data

    # ✅ 당사 특허 요약
    our_patent = get_or_fetch_with_summary(our_patent_id)
    #print("Our Patent Summary:", our_patent["summary"])

    # ✅ 경쟁사 특허 요약
    competitor_patents = []
    for pid in competitor_patent_ids:
        p = get_or_fetch_with_summary(pid)
        competitor_patents.append(p)

    #print("Competitors Patent Summary:")
    #print([p["summary"] for p in competitor_patents])

    # ✅ 포지셔닝 분석 : summary만 전달
    pos_result = analyze_positioning(our_patent["summary"], 
                                     [competitor_patent["summary"] for competitor_patent in competitor_patents])
    print("Positioning Analysis Result:", pos_result)

    # ✅ 구현 차별성 분석 : summary만 전달
    imp_diff_result = analyze_implementation_diff(our_patent["summary"],
                                                  [competitor_patent["summary"] for competitor_patent in competitor_patents])
    

    print("Implementation Difference Analysis Result:", imp_diff_result)

    return pos_result, imp_diff_result