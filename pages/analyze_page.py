import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import seaborn as sns
import numpy as np

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
    st.dataframe(df.head(10), use_container_width=True)

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

if st.button("🚀 Mulai Training", type="primary", use_container_width=True):

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

        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_balanced, y_train_balanced = smote.fit_resample(
            X_train_tfidf, y_train
        )
        

        model = MultinomialNB(alpha=0.1)
        model.fit(X_train_balanced, y_train_balanced)

        y_test_pred = model.predict(X_test_tfidf)
        y_train_pred = model.predict(X_train_tfidf)

        accuracy = accuracy_score(y_test, y_test_pred)

        st.session_state.model = model
        st.session_state.vectorizer = vectorizer

    st.success(f"✅ Training selesai | Akurasi: {accuracy:.2%}")

    # ================= REPORT =================
    st.subheader("📋 Classification Report Test")
    report = classification_report(y_test, y_test_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    st.subheader("📋 Classification Report Train")
    report = classification_report(y_train, y_train_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    # ================= CONFUSION MATRIX IMPROVED =================
    st.subheader("🔲 Confusion Matrix")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric("Akurasi Test Set", f"{accuracy:.2%}", delta=None)
        
    cm = confusion_matrix(y_test, y_test_pred)
    labels = sorted(df['target'].unique())
    
    # Hitung persentase per baris (untuk normalisasi)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure dengan 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Count-based heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Jumlah Prediksi'}, ax=ax1,
                linewidths=0.5, linecolor='gray')
    ax1.set_title('Confusion Matrix (Count)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Actual Label', fontsize=11, fontweight='bold')
    
    # Plot 2: Percentage-based heatmap
    annot_text = np.array([[f'{count}\n({pct:.1f}%)' 
                            for count, pct in zip(row_count, row_pct)]
                           for row_count, row_pct in zip(cm, cm_percent)])
    
    sns.heatmap(cm_percent, annot=annot_text, fmt='', cmap='RdYlGn', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Persentase (%)'}, ax=ax2,
                linewidths=0.5, linecolor='gray', vmin=0, vmax=100)
    ax2.set_title('Confusion Matrix (Normalized %)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Actual Label', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Tambahkan detail metrik per kelas
    with st.expander("📊 Detail Metrik Per Kelas"):
        metrics_per_class = []
        for i, label in enumerate(labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics_per_class.append({
                'Kelas': label,
                'True Positive': tp,
                'False Positive': fp,
                'False Negative': fn,
                'True Negative': tn,
                'Precision': f"{precision:.2%}",
                'Recall': f"{recall:.2%}",
                'F1-Score': f"{f1:.2%}"
            })
        
        metrics_df = pd.DataFrame(metrics_per_class)
        st.dataframe(metrics_df, use_container_width=True)

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

        st.dataframe(proba_df, use_container_width=True)