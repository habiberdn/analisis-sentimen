import streamlit as st
from lib.preprocessing import preprocessing

st.title("Preprocessing")

file_uplaod =st.file_uploader("Uplaod CSV")
st.button("Preprocess!")
