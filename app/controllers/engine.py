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
from app.controllers.implementation import analyze_implementation_diff, analyze_implementation_diff_by_axis

# 미래 확장: positioning, implementation, strategy

    # ✅ 특허 메타데이터 및 요약 가져오기/생성(에러나면 None 반환)
def get_or_fetch_with_summary(patent_id):
    try:
        patent_data = load_cached_patent(patent_id)
        if not patent_data:
            patent_data = fetch_patent_metadata(patent_id)
            if not patent_data:
                raise ValueError("Patent metadata not found")
            save_patent_cache(patent_id, patent_data)

        summary = load_summary(patent_id)
        if summary is None:
            summary = summarize_patent(patent_data["description"], patent_id, "v2")
            if not summary:
                raise ValueError("Summary generation failed")
            save_summary(patent_id, summary)

        patent_data["summary"] = summary
        return patent_data

    except Exception as e:
        print(f"[ERROR] Patent {patent_id} 처리 실패: {e}")
        return None  # 실패 시 None 반환

def run_analysis(our_patent_id: str, competitor_patent_ids: list[str], 
status_text, progress_bar, show_progress) -> dict:

    # ✅ 당사 특허 요약
    show_progress("fetch", status_text, progress_bar)
    our_patent = get_or_fetch_with_summary(our_patent_id)
    #print("Our Patent Summary:", our_patent["summary"])

    # ✅ 경쟁사 특허 요약
    show_progress("summarize", status_text, progress_bar)
    competitor_patents = []
    for pid in competitor_patent_ids:
        p = get_or_fetch_with_summary(pid)
        competitor_patents.append(p)

    #print("Competitors Patent Summary:")
    #print([p["summary"] for p in competitor_patents])

    # ✅ 포지셔닝 분석 : summary만 전달
    show_progress("positioning", status_text, progress_bar)
    pos_result = analyze_positioning(our_patent["summary"], 
                                     [competitor_patent["summary"] for competitor_patent in competitor_patents])
    print("Positioning Analysis Result:", pos_result)

    show_progress("diff", status_text, progress_bar)
    # ✅ 구현 차별성 분석 : summary만 전달(1:1)
    imp_diff_result = analyze_implementation_diff(our_patent["summary"],
                                                  [competitor_patent["summary"] for competitor_patent in competitor_patents])
    
    # 구현 차별성 분석 : 1:N 요약
    imp_diff_by_axis = analyze_implementation_diff_by_axis(imp_diff_result, our_patent_id)


    print("Implementation Difference Analysis Result:", imp_diff_by_axis)

    return pos_result, imp_diff_result, imp_diff_by_axis