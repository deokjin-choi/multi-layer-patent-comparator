import streamlit as st
import time

def show_progress():
    st.subheader("🔄 Running 3-Layer Patent Analysis...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(1, 101):
        progress_bar.progress(i)
        status_text.text(f"Step {i}/100")
        time.sleep(0.01)  # 이건 실제 처리 대체용입니다.

    st.success("✅ Analysis complete!")