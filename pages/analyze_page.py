import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🔍 Klasifikasi Teks - Multinomial Naive Bayes")
st.markdown("---")

df = st.session_state.get("processed_data", None)

if df is None:
    st.warning("⚠️ Data belum tersedia.")
    st.stop()

st.subheader("📊 Informasi Dataset")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Data", len(df))
with col2:
    st.metric("Jumlah Kolom", len(df.columns))
with col3:
    st.metric("Jumlah Kelas", df['target'].nunique())

with st.expander("🔍 Preview Data"):
    st.dataframe(df.head(10), width="stretch")

st.subheader("📊 Pembagian Data")
split_option = st.radio(
    "Rasio Train : Test",
    ["70:30", "80:20", "90:10"],
    index=1,
    horizontal=True
)
train_size, test_size = map(int, split_option.split(':'))

with st.expander("🔧 Parameter TF-IDF"):
    max_features = st.number_input(
        "Max Features",
        min_value=1000,
        max_value=15000,
        value=3000,
        step=500
    )

if st.button("🚀 Mulai Training", type="primary", width="stretch"):

    with st.spinner("⏳ Training..."):

        X = df['stemmed'].astype(str)
        y = df['target']

        # Safety check
        if X.str.strip().eq("").any():
            st.error("❌ Masih ada teks kosong setelah preprocessing")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size / 100,
            random_state=42,
            stratify=y
        )

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            token_pattern=r'(?u)\b\w+\b',
            min_df=3,
            max_df=0.8,
        )

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        model = MultinomialNB(alpha=1.5)
        model.fit(X_train_tfidf, y_train)

        y_test_pred = model.predict(X_test_tfidf)
        y_train_pred = model.predict(X_train_tfidf)

        accuracy = accuracy_score(y_test, y_test_pred)

        st.session_state.model = model
        st.session_state.vectorizer = vectorizer

    st.success(f"✅ Training selesai | Akurasi: {accuracy:.2%}")

    # ================= REPORT =================
    st.subheader("📋 Classification Report Test")
    report = classification_report(y_test, y_test_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose(), width="stretch")

    st.subheader("📋 Classification Report Train")
    report = classification_report(y_train, y_train_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose(), width="stretch")

    st.subheader("🔲 Confusion Matrix")
    st.write(f"Akurasi : {accuracy:.2%}")
    cm = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

if 'model' in st.session_state:
    st.markdown("---")
    st.subheader("🧪 Testing Manual")

    text = st.text_area("Masukkan teks")

    if st.button("Prediksi"):
        vec = st.session_state.vectorizer.transform([text])
        pred = st.session_state.model.predict(vec)[0]
        proba = st.session_state.model.predict_proba(vec)[0]
 
        st.success(f"Hasil Prediksi: **{pred}**")

        proba_df = pd.DataFrame({
            "Kelas": st.session_state.model.classes_,
            "Probabilitas": proba
        }).sort_values("Probabilitas", ascending=False)

        st.dataframe(proba_df, width="stretch")
