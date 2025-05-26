import streamlit as st

def render_implementation_tree(axis_summary: list, our_id: str):
    st.markdown(f"""
    <div style="background-color:#e6f0ff; border-left: 5px solid #0059b3; border-radius: 6px;
                padding: 10px 15px 5px 15px; margin-bottom: 0px;">
        <strong>Implementation-level Comparison:</strong><br>
        Our Patent <code>{our_id}</code> vs multiple competitor patents, grouped by technical category.
    </div>
    <div style="background-color:#f2f2f2; padding: 15px 25px 10px 25px; border-radius: 0px 0px 6px 6px;
                margin-top: 0px;">
    """, unsafe_allow_html=True)

    for item in axis_summary:
        category = item["category"]
        our_text = item["our_summary"]

        # 카테고리별 블록
        block_html = f"""
        <div style='margin-bottom: 10px; line-height: 1.6;'>
            <div style='margin-bottom: 3px;'><strong>├── {category}</strong></div>
            <div style='margin-bottom: 8px;'><strong>│     └─ {our_text}</strong></div>
        """

        # 경쟁사 비교 항목 (▶ 제거됨)
        for comp in item["comparisons"]:
            comp_id = comp["competitor_id"]
            diff = comp["difference"]
            block_html += (
                f"<div style='margin-bottom: 5px;'>"
                f"│        vs {comp_id}: {diff}"
                f"</div>"
            )

        block_html += "</div>"

        st.markdown(block_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
