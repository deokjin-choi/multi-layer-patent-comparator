import streamlit as st
import pickle
import io
import os
from frontend.components.report_generator import generate_ppt_from_result

st.set_page_config(page_title="Download PPT Report", layout="centered")
st.title("📥 Download Patent Comparison Report")

result_path = "data/final_results/last_result.pkl"

if not os.path.exists(result_path):
    st.warning("❌ 분석 결과가 없습니다. 먼저 특허 비교를 실행해 주세요.")
else:
    with open(result_path, "rb") as f:
        result = pickle.load(f)

    ppt_io = io.BytesIO()
    generate_ppt_from_result(
        strategy_output=result["strategy_output"],
        pos_result=result["pos_result"],
        imp_diff_result=result["imp_diff_result"],
        output_path=None,
        out_stream=ppt_io
    )
    ppt_io.seek(0)

    st.download_button(
        label="⬇️ Click to Download PPT Report",
        data=ppt_io,
        file_name="Patent_Comparison_Report.pptx",
        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )
