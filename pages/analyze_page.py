import streamlit as st
import pandas as pd
from typing import cast
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import seaborn as sns
import numpy as np

def calculate_naive_bayes_manual(X_train, y_train, X_test, y_test, count_vectorizer, tf_count_df_train, tf_count_df_test):
    """
    Menghitung Naive Bayes secara manual dengan rumus BENAR:
    P(word|Class) = (count(word,Class) + α) / (Σ count(word',Class) + α × |V|)
    Posterior = Prior × Likelihood

    Dalam log:
    log(Posterior) = log(Prior) + Σ count(word) × log(Likelihood)
    """

    # 1. Prior Probability P(Class)
    class_counts = y_train.value_counts().sort_index()
    total_docs = len(y_train)
    prior = class_counts / total_docs

    st.write("### 1️⃣ Prior Probability P(Class)")
    st.write("**Rumus:** P(Class) = Jumlah dokumen kelas / Total dokumen")
    prior_df = pd.DataFrame({
        'Class': prior.index,
        'Count': class_counts.values,
        'Prior P(Class)': prior.values
    })
    st.dataframe(prior_df)

    # 2. Likelihood P(word|class) dengan Laplace Smoothing - RUMUS BENAR
    alpha = 1.0  # Laplace smoothing (standar = 1)
    classes = sorted(y_train.unique())
    vocabulary_size = len(count_vectorizer.get_feature_names_out())

    st.write("### 2️⃣ Likelihood P(word|Class) dengan Laplace Smoothing")
    st.write("**Rumus:** P(word|Class) = (count(word,Class) + α) / (Σ count(word',Class) + α × |V|)")
    st.write(f"**Parameter:** α (alpha) = {alpha}, |V| (vocabulary size) = {vocabulary_size}")

    likelihood = {}
    likelihood_info = []

    for c in classes:
        idx = y_train == c

        # Hitung total kemunculan setiap kata dalam kelas c (TANPA smoothing dulu)
        raw_word_count = tf_count_df_train.loc[idx].sum(axis=0)
        # Total kata dalam kelas c (dengan smoothing)
        # RUMUS BENAR: Σ count(word',C) + α × |V|
        total_words_in_class = raw_word_count.sum() + (alpha * vocabulary_size)

        # Likelihood untuk setiap kata
        # RUMUS BENAR: (count(word,C) + α) / (Σ count(word',C) + α × |V|)
        likelihood[c] = (raw_word_count + alpha) / total_words_in_class

        likelihood_info.append({
            'Class': c,
            'Σ count(word\',C) [raw]': raw_word_count.sum(),
            'α × |V|': alpha * vocabulary_size,
            'Total [denominator]': total_words_in_class
        })

    st.write("**Perhitungan denominator untuk setiap kelas:**")
    st.dataframe(pd.DataFrame(likelihood_info))

    likelihood_df = pd.DataFrame(likelihood)
    st.write("**Likelihood P(word|Class) untuk setiap term:**")
    st.dataframe(likelihood_df)

    # Verifikasi: sum dari semua likelihood harus = 1 untuk setiap class
    st.write("**Verifikasi:** Σ P(word|Class) untuk setiap kelas (harus ≈ 1.0):")
    verification = likelihood_df.sum()
    st.dataframe(verification.rename("Sum of Likelihoods").reset_index().rename(columns={"index": "Class"}))

    # 3. Posterior Probability untuk TRAINING data
    st.write("### 3️⃣ Posterior Probability (TRAINING)")
    st.write("**Rumus:** log P(Class|Doc) = log P(Class) + Σ count(word) × log P(word|Class)")

    posteriors_train = []
    posterior_details = []

    for idx, row in tf_count_df_train.iterrows():
        doc_post = {}
        doc_detail = {'doc_index': idx}

        for c in classes:
            # log(Prior)
            log_prior = np.log(prior[c])
            # log(Likelihood) untuk setiap kata
            log_likelihood = np.log(likelihood[c])
            # Posterior = Prior × ∏ Likelihood^count
            # Log Posterior = log(Prior) + Σ count × log(Likelihood)
            log_likelihood_sum = (row * log_likelihood).sum()
            log_posterior = log_prior + log_likelihood_sum

            doc_post[c] = log_posterior
            doc_detail[f'log_prior_C{c}'] = log_prior
            doc_detail[f'sum_log_lik_C{c}'] = log_likelihood_sum
            doc_detail[f'log_post_C{c}'] = log_posterior

        posteriors_train.append(doc_post)
        posterior_details.append(doc_detail)

    posterior_df_train = pd.DataFrame(posteriors_train)
    st.write("**Log Posterior untuk setiap dokumen training:**")
    st.dataframe(posterior_df_train)

    with st.expander("📊 Detail Perhitungan Posterior"):
        st.dataframe(pd.DataFrame(posterior_details))

    # 4. Prediksi TRAINING
    pred_manual_train = posterior_df_train.idxmax(axis=1).values
    accuracy_train = (pred_manual_train == y_train.values).mean()

    st.write("### 4️⃣ Prediksi pada Data TRAINING")
    train_results = pd.DataFrame({
        "Teks": X_train.values,
        "Label Sebenarnya": y_train.values,
        "Prediksi": pred_manual_train,
        "Benar": pred_manual_train == y_train.values
    })
    st.dataframe(train_results)

    # 5. Posterior Probability untuk TESTING data
    st.write("### 5️⃣ Posterior Probability (TESTING)")
    st.write("Menggunakan Prior dan Likelihood yang sama dari data training")

    posteriors_test = []
    for idx, row in tf_count_df_test.iterrows():
        doc_post = {}
        for c in classes:
            log_prior = np.log(prior[c])
            log_likelihood = np.log(likelihood[c])
            log_posterior = log_prior + (row * log_likelihood).sum()
            doc_post[c] = log_posterior
        posteriors_test.append(doc_post)

    posterior_df_test = pd.DataFrame(posteriors_test)
    st.write("**Log Posterior untuk setiap dokumen testing:**")
    st.dataframe(posterior_df_test)

    # 6. Prediksi TESTING
    pred_manual_test = posterior_df_test.idxmax(axis=1).values
    accuracy_test = (pred_manual_test == y_test.values).mean()

    st.write("### 6️⃣ Prediksi pada Data TESTING")
    test_results = pd.DataFrame({
        "Teks": X_test.values,
        "Label Sebenarnya": y_test.values,
        "Prediksi": pred_manual_test,
        "Benar": pred_manual_test == y_test.values
    })
    st.dataframe(test_results)

    return pred_manual_train, pred_manual_test

st.title("🔍 Klasifikasi Teks - Multinomial Naive Bayes")
st.markdown("---")

df = st.session_state.get("processed_data", None)

if df is None:
    st.warning("⚠️ Data belum tersedia.")
    st.stop()

st.subheader("📊 Informasi Dataset")

# Hitung statistik frasa
has_phrases_count = 0
total_phrases = 0
if 'found_phrases' in df.columns:
    has_phrases_count = df['found_phrases'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()
    total_phrases = df['found_phrases'].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Data", len(df))
with col2:
    st.metric("Jumlah Kolom", len(df.columns))
with col3:
    st.metric("Jumlah Kelas", df['target'].nunique())
with col4:
    if 'found_phrases' in df.columns:
        st.metric("Data dengan Frasa", f"{has_phrases_count} ({has_phrases_count/len(df)*100:.1f}%)")

# ================= PREVIEW DATA DENGAN FILTER FRASA =================
with st.expander("🔍 Preview Data", expanded=False):

    # Tab untuk filter
    tab1, tab2, tab3 = st.tabs(["📋 Semua Data", "🔤 Data dengan Frasa", "📊 Analisis Frasa"])

    with tab1:
        st.dataframe(df.drop(columns=['found_phrases','found_single_words']).head(10), use_container_width=True)

    with tab2:
        if 'found_phrases' in df.columns:
            # Filter data yang memiliki frasa
            df_with_phrases = df[df['found_phrases'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)].copy()

            if len(df_with_phrases) > 0:
                st.info(f"📊 Ditemukan {len(df_with_phrases)} data yang mengandung frasa")

                # Pilih kolom yang akan ditampilkan
                display_cols = ['full_text', 'normalized', 'lexicon_score', 'target', 'found_phrases']
                available_cols = [col for col in display_cols if col in df_with_phrases.columns]

                # Format dataframe untuk tampilan yang lebih baik
                df_display = df_with_phrases[available_cols].head(20).copy()

                # Format kolom found_phrases agar lebih readable
                def format_phrases(phrases_list):
                    if not isinstance(phrases_list, list) or len(phrases_list) == 0:
                        return "—"

                    formatted = []
                    for p in phrases_list:
                        if isinstance(p, dict):
                            text = p.get('text', '')
                            weight = p.get('weight', 0)
                            emoji = "✅" if weight > 0 else "❌" if weight < 0 else "➖"
                            formatted.append(f"{emoji} {text} ({weight:+d})")
                    return " | ".join(formatted)

                df_display['found_phrases'] = df_display['found_phrases'].apply(format_phrases)

                # Styling untuk sentiment
                def highlight_sentiment(row):
                    if row['target'] == 'positif':
                        return ['background-color: #00e836'] * len(row)
                    elif row['target'] == 'negatif':
                        return ['background-color: #fc0015'] * len(row)
                    else:
                        return ['background-color: #ffbf00'] * len(row)

                st.dataframe(
                    df_display.style.apply(highlight_sentiment, axis=1),
                    use_container_width=True,
                    height=400
                )

            else:
                st.warning("Tidak ada data yang mengandung frasa")
        else:
            st.info("Kolom 'found_phrases' tidak tersedia. Pastikan preprocessing sudah menggunakan versi terbaru.")

    with tab3:
        if 'found_phrases' in df.columns:
            st.markdown("### 📊 Statistik Frasa dalam Dataset")

            col_stat1, col_stat2, col_stat3 = st.columns(3)

            with col_stat1:
                st.metric("Total Frasa Ditemukan", total_phrases)

            with col_stat2:
                avg_phrases = df['found_phrases'].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
                st.metric("Rata-rata Frasa per Teks", f"{avg_phrases:.2f}")

            with col_stat3:
                pct_with_phrases = (has_phrases_count / len(df)) * 100
                st.metric("% Data dengan Frasa", f"{pct_with_phrases:.1f}%")

            # Top frasa yang paling sering muncul
            st.markdown("#### 🏆 Top 10 Frasa Paling Sering Muncul")

            from collections import Counter
            all_phrases = []

            for phrases in df['found_phrases']:
                if isinstance(phrases, list):
                    for p in phrases:
                        if isinstance(p, dict):
                            all_phrases.append(p.get('text', ''))

            phrase_counts = Counter(all_phrases)

            if phrase_counts:
                top_phrases = pd.DataFrame(
                    phrase_counts.most_common(10),
                    columns=['Frasa', 'Frekuensi']
                )

                # Tambahkan informasi bobot dari lexicon jika tersedia
                if 'lexicon' in dir():
                    from lib.preprocessing import lexicon
                    top_phrases['Bobot'] = top_phrases['Frasa'].apply(
                        lambda x: lexicon.get(x, 0)
                    )

                st.dataframe(top_phrases, use_container_width=True, hide_index=True)

                # Visualisasi
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(top_phrases['Frasa'], top_phrases['Frekuensi'], color='steelblue')
                ax.set_xlabel('Frekuensi', fontweight='bold')
                ax.set_ylabel('Frasa', fontweight='bold')
                ax.set_title('Top 10 Frasa Paling Sering Muncul', fontweight='bold', pad=15)
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Tidak ada frasa yang ditemukan dalam dataset")

            # Distribusi frasa per sentiment
            st.markdown("#### 📈 Distribusi Frasa per Sentimen")

            sentiment_phrase_stats = df.groupby('target')['found_phrases'].apply(
                lambda x: x.apply(lambda p: len(p) if isinstance(p, list) else 0).sum()
            )

            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                fig, ax = plt.subplots(figsize=(8, 6))
                sentiment_phrase_stats.plot(kind='bar', ax=ax, color=['#f8d7da', '#fff3cd', '#d4edda'])
                ax.set_title('Total Frasa per Sentimen', fontweight='bold', pad=15)
                ax.set_xlabel('Sentimen', fontweight='bold')
                ax.set_ylabel('Jumlah Frasa', fontweight='bold')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                plt.tight_layout()
                st.pyplot(fig)

            with col_chart2:
                # Persentase data dengan frasa per sentiment
                phrase_presence = df.groupby('target').apply(
                    lambda x: (x['found_phrases'].apply(lambda p: len(p) > 0 if isinstance(p, list) else False).sum() / len(x)) * 100
                )

                fig, ax = plt.subplots(figsize=(8, 6))
                phrase_presence.plot(kind='bar', ax=ax, color=['#f8d7da', '#fff3cd', '#d4edda'])
                ax.set_title('% Data dengan Frasa per Sentimen', fontweight='bold', pad=15)
                ax.set_xlabel('Sentimen', fontweight='bold')
                ax.set_ylabel('Persentase (%)', fontweight='bold')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                ax.set_ylim(0, 100)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("Kolom 'found_phrases' tidak tersedia")

# ================= PEMBAGIAN DATA =================
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

        # Simpan model dan vectorizer ke session state
        st.session_state.model = model
        st.session_state.vectorizer = vectorizer

    st.success(f"✅ Training selesai | Akurasi: {accuracy:.2%}")

    # ================= PERHITUNGAN MANUAL NAIVE BAYES (10 TWEET PERTAMA) =================
    with st.expander("📐 Perhitungan Manual Naive Bayes", expanded=False):
        st.info("📌 Perhitungan manual ini menggunakan 10 tweet pertama dari dataset asli (8 training + 2 testing)")
        st.markdown("---")

        st.markdown("""
        **Rumus Naive Bayes Multinomial:**
        - P(word|Class) = (count(word,Class) + α) / (Σ count(word',Class) + α × |V|)
        - P(Class|Doc) ∝ P(Class) × ∏ P(word|Class)^count(word)
        - Log form: log P(Class|Doc) = log P(Class) + Σ count(word) × log P(word|Class)
        """)

        # Ambil 10 data PERTAMA dari dataset asli (sebelum split)
        df_10_first = df.head(10).copy()
        X_10_all = df_10_first['stemmed'].astype(str).reset_index(drop=True)
        y_10_all = df_10_first['target'].reset_index(drop=True)

        # Split manual: 8 untuk training, 2 untuk testing
        X_train_10 = X_10_all.head(8).reset_index(drop=True)
        y_train_10 = y_10_all.head(8).reset_index(drop=True)
        X_test_2 = X_10_all.tail(2).reset_index(drop=True)
        y_test_2 = y_10_all.tail(2).reset_index(drop=True)

        st.write("**Data Training (8 tweet pertama dari dataset):**")
        preview_df_train = pd.DataFrame({
            'Index': range(1, 9),
            'Teks': X_train_10.values,
            'Label': y_train_10.values
        })
        st.dataframe(preview_df_train, use_container_width=True)

        st.write("**Data Testing (2 tweet berikutnya dari dataset):**")
        preview_df_test = pd.DataFrame({
            'Index': range(9, 11),
            'Teks': X_test_2.values,
            'Label': y_test_2.values
        })
        st.dataframe(preview_df_test, use_container_width=True)

        st.markdown("---")
        st.write("## 🔢 Feature Extraction dengan Count Vectorizer")
        st.write("Naive Bayes Multinomial menggunakan **count** (bukan TF-IDF)")

        # Count Vectorizer untuk 10 data pertama
        count_vectorizer = CountVectorizer(
            ngram_range=(1, 1),
            token_pattern=r'(?u)\b\w+\b',
            min_df=1,
            max_df=1.0
        )

        X_count_train_10 = count_vectorizer.fit_transform(X_train_10)
        X_count_test_2 = count_vectorizer.transform(X_test_2)

        tf_count_df_train_10 = pd.DataFrame(
            X_count_train_10.toarray(),
            columns=count_vectorizer.get_feature_names_out()
        ).reset_index(drop=True)
        tf_count_df_test_2 = pd.DataFrame(
            X_count_test_2.toarray(),
            columns=count_vectorizer.get_feature_names_out()
        ).reset_index(drop=True)

        st.write("**Term Frequency (Count) - Training (10 tweets):**")
        st.dataframe(tf_count_df_train_10, use_container_width=True)

        st.write("**Term Frequency (Count) - Testing (2 tweets):**")
        st.dataframe(tf_count_df_test_2, use_container_width=True)

        st.markdown("---")
        st.write("## 🧮 Perhitungan Naive Bayes Manual")

        # Hitung Naive Bayes manual
        calculate_naive_bayes_manual(
            X_train_10, y_train_10, X_test_2, y_test_2,
            count_vectorizer, tf_count_df_train_10, tf_count_df_test_2
        )

    # ================= REPORT =================
    with st.expander("📊 Classification Report"):
        tab1, tab2 = st.tabs(["Testing", "Training"])
        with tab1:
            report = classification_report(y_test, y_test_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
        with tab2:
            report = classification_report(y_train, y_train_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    # ================= CONFUSION MATRIX IMPROVED =================
    st.subheader("🔲 Confusion Matrix")


    cm = confusion_matrix(y_test, y_test_pred)
    labels = sorted(df['target'].unique())

    # Hitung persentase per baris (untuk normalisasi)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create figure dengan 1 subplot
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 6))

    # Plot: Count-based heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Jumlah Prediksi'}, ax=ax1,
                linewidths=0.5, linecolor='gray')
    ax1.set_title('Confusion Matrix (Count)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Actual Label', fontsize=11, fontweight='bold')

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

# ================= TESTING MANUAL =================
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
