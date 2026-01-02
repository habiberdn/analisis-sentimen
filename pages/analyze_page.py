import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🔍 Klasifikasi Teks dengan Multinomial Naive Bayes")
st.markdown("---")

# Cek apakah data sudah tersedia di session state
if 'processed_data' not in st.session_state :
    st.session_state.processed_data = None

df = st.session_state.processed_data

if df is None:
    st.warning("⚠️ Data belum tersedia. Silakan upload dan proses data terlebih dahulu.")

else :
    # Tampilkan info dataset
    st.subheader("📊 Informasi Dataset")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Data", len(df))
    with col2:
        st.metric("Jumlah Kolom", len(df.columns))
    with col3:
        if 'label' in df.columns or 'target' in df.columns:
            label_col = 'label' if 'label' in df.columns else 'target'
            st.metric("Jumlah Kelas", df[label_col].nunique())

    # Preview data
    with st.expander("Lihat Data"):
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    st.subheader("📝 Pilih Kolom Data")
    col1, col2 = st.columns(2)
    with col1:
        text_column = st.selectbox(
            "Kolom Teks",
            options=df.columns.tolist(),
            help="Pilih kolom yang berisi teks untuk klasifikasi"
        )

    with col2:
        # Cari kolom label secara otomatis
        default_label = None
        if 'label' in df.columns:
            default_label = df.columns.tolist().index('label')
        elif 'target' in df.columns:
            default_label = df.columns.tolist().index('target')

        label_column = st.selectbox(
            "Kolom Label",
            options=df.columns.tolist(),
            index=default_label if default_label is not None else 0,
            help="Pilih kolom yang berisi label/target klasifikasi"
        )

    st.markdown("---")

    # Pilihan rasio train-test
    st.subheader("📊 Pembagian Data")
    split_option = st.radio(
        "Pilih Rasio Training : Testing",
        options=["70:30", "80:20", "90:10"],
        index=1, # 80:20
        horizontal=True,
        help="Rasio pembagian data training dan testing"
    )

    train_size, test_size = map(int, split_option.split(':'))

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"🎓 Training: {train_size}%")
    with col2:
        st.info(f"🧪 Testing: {test_size}%")

    # Parameter TF-IDF
    with st.expander("🔧 Parameter TF-IDF (Opsional)"):
        col1, col2 = st.columns(2)
        with col1:
            max_features = st.number_input(
                "Max Features",
                min_value=1000,
                max_value=10000,
                value=5000,
                step=1000,
                help="Jumlah maksimum fitur yang akan digunakan"
            )

    st.markdown("---")

    # Tombol untuk memulai training
    if st.button("🚀 Mulai Training & Testing", type="primary", use_container_width=True):
        try:
            with st.spinner("⏳ Sedang melakukan training dan testing..."):
                # Ambil data
                X = df[text_column].astype(str) # fitur (input)
                y = df[label_column] # Menentukan kolom target/kelas yang ingin diprediksi (output)

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size/100,
                    random_state=42,
                    stratify=y
                )

                # TF-IDF Vectorization
                vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1,2)
                )
                X_train_tfidf = vectorizer.fit_transform(X_train)
                X_test_tfidf = vectorizer.transform(X_test)

                # Training Multinomial Naive Bayes
                model = MultinomialNB()
                model.fit(X_train_tfidf, y_train)

                # Prediksi
                y_pred = model.predict(X_test_tfidf)

                # Hitung akurasi
                accuracy = accuracy_score(y_test, y_pred)

                # Simpan model ke session state
                st.session_state.model = model
                st.session_state.vectorizer = vectorizer
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred

            st.success(f"✅ Training selesai! Akurasi: {accuracy:.2%}")

            # Tampilkan hasil
            st.markdown("---")
            st.subheader("📈 Hasil Evaluasi")

            # Metrik utama
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Akurasi", f"{accuracy:.2%}")
            with col2:
                st.metric("Data Training", len(X_train))
            with col3:
                st.metric("Data Testing", len(X_test))

            # Classification Report
            st.subheader("📋 Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']),
                         use_container_width=True)
            # Confusion Matrix
            st.subheader("🔲 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)

            # Distribusi prediksi
            st.subheader("📊 Distribusi Prediksi")
            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                pd.Series(y_test).value_counts().plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title('Distribusi Label Aktual (Testing)')
                ax.set_xlabel('Label')
                ax.set_ylabel('Jumlah')
                plt.xticks(rotation=45)
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                pd.Series(y_pred).value_counts().plot(kind='bar', ax=ax, color='lightcoral')
                ax.set_title('Distribusi Label Prediksi')
                ax.set_xlabel('Label')
                ax.set_ylabel('Jumlah')
                plt.xticks(rotation=45)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {str(e)}")
            st.exception(e)

    # Bagian untuk testing manual
    if 'model' in st.session_state and 'vectorizer' in st.session_state:
        st.markdown("---")
        st.subheader("🧪 Testing Manual")

        test_text = st.text_area(
            "Masukkan teks untuk diprediksi:",
            height=100,
            placeholder="Ketik teks di sini..."
        )

        if st.button("Prediksi", use_container_width=True):
            if test_text:
                try:
                    # Transform dan prediksi
                    text_tfidf = st.session_state.vectorizer.transform([test_text])
                    prediction = st.session_state.model.predict(text_tfidf)[0]
                    proba = st.session_state.model.predict_proba(text_tfidf)[0]

                    st.success(f"**Hasil Prediksi:** {prediction}")

                    # Tampilkan probabilitas
                    st.write("**Probabilitas untuk setiap kelas:**")
                    proba_df = pd.DataFrame({
                        'Kelas': st.session_state.model.classes_,
                        'Probabilitas': proba
                    }).sort_values('Probabilitas', ascending=False)

                    st.dataframe(proba_df, use_container_width=True)

                    # Bar chart probabilitas
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.barh(proba_df['Kelas'].astype(str), proba_df['Probabilitas'])
                    ax.set_xlabel('Probabilitas')
                    ax.set_title('Probabilitas Prediksi per Kelas')
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error saat prediksi: {str(e)}")
            else:
                st.warning("Silakan masukkan teks terlebih dahulu.")
