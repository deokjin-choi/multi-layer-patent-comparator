import streamlit as st

def render():
    st.subheader("Enter Patent Numbers")
    
    col1, col2 = st.columns(2)
    with col1:
        our_patent = st.text_input("Your Company’s Patent", placeholder="e.g., US1234567B2")
    with col2:
        competitor_patents = st.text_area("Competitor Patents (multiple allowed)", height=100,
                                          placeholder="Separate by commas or new lines")

    submitted = st.button("Run Analysis")

    return {
        "our_patent": our_patent.strip(),
        "competitor_patents": [p.strip() for p in competitor_patents.replace(',', '\n').splitlines() if p.strip()],
        "submitted": submitted
    }
