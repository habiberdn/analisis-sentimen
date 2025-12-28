import streamlit as st
from lib.scrape import download

@st.dialog("Combine CSV")
def combine():
    firstCSV = st.file_uploader("First CSV", key="first_csv")
    secondCSV = st.file_uploader("Second CSV", key="second_csv")

    if st.button("Combine & Download"):
        if firstCSV is None or secondCSV is None:
            st.warning("Upload kedua file CSV terlebih dahulu")
            return
        csv_data = download(firstCSV, secondCSV)
        st.success("CSV berhasil digabung")
        st.download_button(
            label="Download Combined CSV",
            data=csv_data,
            file_name="combined.csv",
            mime="text/csv",
            use_container_width=True
        )

def render_sidebar(authenticator):
    """Render sidebar yang sama di semua halaman"""
    with st.sidebar:
        # Buttons
        col1, col2 = st.columns(2, gap="small")

        with col1:
            if st.button("Combine CSV", use_container_width=True):
                combine()

        with col2:
            if authenticator.logout(use_container_width=True):
                st.session_state.clear()
                st.rerun()
