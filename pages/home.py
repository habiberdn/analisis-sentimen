import streamlit as st


st.markdown("""
<style>
.center-container {
    background-color: #0d243d;
    padding: 40px;
    border-radius: 20px;
    text-align: center;
    display : flex;
    flex-direction : column;
    gap: 25px;
    /* center secara horizontal */
    max-width: 700px;
    margin: auto;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="center-container">
    <h2>Selamat Datang Pada Aplikasi Analisis Sentimen</h2>
    <p>Pengembangan applikasi ini dilakukan untuk menghasilkan sebuah sentimen dari suatu opini berdasarkan suatu isu atau kasus pada sosial media X. Opini tersebut akan dianalisis menggunakan pendekatan teknik Natural Language Processing dengan metode Naive Bayes Classifier.</p>
</div>
""", unsafe_allow_html=True)
