import streamlit as st
from lib.scrape import scrape_tweets

if "disabled" not in st.session_state:
    st.session_state.disabled = True

if "df" not in st.session_state:
    st.session_state.df = None

st.title("Halaman Scrapping")

auth_token = st.text_input("Twitter Auth Token", type="password")
keyword = st.text_input("Keyword")
limit = st.number_input("Limit", min_value=1, value=100)

col1, col2  = st.columns(2, gap="small")

csv_data = st.session_state.df.to_csv(index=False) if st.session_state.df is not None else ""

with col1:
    scrape_btn = st.button("Scrape",use_container_width=True)
with col2:
    export_btn = st.download_button(label="Download as CSV",use_container_width=True,disabled=st.session_state.disabled,file_name="cek_kesehatan_gratis.csv",
    mime="text/csv", data=csv_data)

if scrape_btn:
    with st.spinner("Mengambil data..."):
        df = scrape_tweets(auth_token, keyword, limit)
    st.session_state.disabled = False
    st.session_state.df = df
    st.success(f"Berhasil mengambil {len(df)} tweet")
    st.dataframe(df)
    st.rerun()
