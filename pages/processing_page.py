from logging import exception
import streamlit as st
from lib.preprocessing import *
import pandas as pd

st.title("Text Preprocessing")

# Initialization
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None

file_upload =st.file_uploader("Uplaod CSV")

# Check file is upload
if file_upload is not None:
    st.session_state['uploaded_file'] = file_upload

# Preview Data
if st.session_state['uploaded_file'] is not None:
    try:
        data = read_data(st.session_state['uploaded_file'])
        with st.expander("👀 Preview Data (5 Baris Pertama)"):
            st.dataframe(data, use_container_width=True)

        # Mengembalikan posisi pointer ke posisi awal agar bisa membaca file
        st.session_state['uploaded_file'].seek(0)
    except Exception as e:
        st.warning(f"Could not preview data: {e}")

    # Preprocessing action
    if st.button("Mulai Preprocessing"):
        try:
            with st.spinner("Loading data..."):
                data = read_data(st.session_state['uploaded_file'])
            if 'full_text' not in data.columns:
                st.error(f"❌ Kolom 'full_text' Tidak Ditemukan!")
                st.info(f"Kolom yang tersedia: {', '.join(data.columns)}")
                st.stop()
            st.success(f"✅ Memuat {len(data):,} Baris")
            # Show processing animation
            with st.spinner("🔄 Processing data using optimized batch method..."):
                # Progress container
                progress_container = st.container()

                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Use the optimized batch function
                    status_text.text("⚡ Running batch preprocessing (vectorized operations)...")

                    # Update progress manually during processing
                    progress_bar.progress(0.1)

                    # Call the optimized batch function
                    processed_data = preprocessing_batch(
                        df=data,
                        text_column='full_text'
                    )

                    progress_bar.progress(1.0)
                    status_text.text("✅ Batch processing complete!")

                st.session_state['processed_data'] = processed_data
                st.success("Text preprocessing completed successfully!")

        except Exception as e:
            st.error(f"❌ Error during preprocessing: {str(e)}")
            import traceback
            with st.expander("🔍 Show detailed error"):
                st.code(traceback.format_exc())


if st.session_state['processed_data'] is not None:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Cleaned Data", "Normalize", "Tokenized", "Stemming","Stopword", "Overview"])
    data = st.session_state["processed_data"]
    st.write(data)
    with tab1:
        st.dataframe(
            data[['full_text', 'cleaned']].head(20),
            use_container_width=True
        )
    with tab2:
        st.dataframe(
            data[['full_text', 'normalized']].head(20),
            use_container_width=True
        )
    with tab3:
        st.dataframe(
            data[['full_text', 'tokenized']].head(20),
            use_container_width=True
        )
    with tab4:
        st.dataframe(
            data[['full_text', 'stopword']].head(20),
            use_container_width=True
        )
    with tab5:
        st.dataframe(
            data[['full_text', 'stemmed']].head(20),
            use_container_width=True
        )
else:
    st.markdown("""
    ### 📋 Requirements:
    - CSV file must contain a column named **`full_text`**
    - Text should be in Indonesian language
    - Supported formats: `.csv`

    ### 🔄 Processing Steps:
    1. **Cleaning**: Remove URLs, mentions, special characters
    2. **Normalization**: Convert slang to formal words
    3. **Tokenization**: Split text into words
    4. **Stemming**: Remove affixes
    5. **Stopwords Removal**: Remove common words
    """)
