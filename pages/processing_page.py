
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

                    # Update progress manually during processing
                    progress_bar.progress(0.1)

                    # Call the optimized batch function
                    processed_data = preprocessing_batch(
                        df=data,
                        text_column='full_text'
                    )


                    progress_bar.progress(1.0)

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
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔴 Original**")
            st.dataframe(
                data[['full_text']].head(10),
                use_container_width=True,
                height=400
            )
        with col2:
            st.markdown("**🟢 Cleaned**")
            st.dataframe(
                data[['cleaned']].head(10),
                use_container_width=True,
                height=400
            )

    with tab2:
        # Mengambil text yang berubah akibat normalisasi
        normalized_texts = data[data['cleaned'] != data['normalized']]

        if len(normalized_texts) > 0:
            st.success(f"✅ Menemukan {len(normalized_texts):,} texts dengan slang yang dinormalisasi")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Sebelum Normalization**")
                st.dataframe(
                    normalized_texts[['cleaned']].head(10),
                    use_container_width=True,
                    height=400
                )
            with col2:
                st.markdown("**Setelah Normalization**")
                st.dataframe(
                    normalized_texts[['normalized']].head(10),
                    use_container_width=True,
                    height=400
                )
        else:
            st.info("Tidak ada perubahaan akibat normalisasi")

    with tab3:
        # Calculate token statistics
        data['token_count'] = data['tokenized'].str.len()

        col1, col2 = st.columns(2)
        with col1:
            avg_tokens = data['token_count'].mean()
            st.metric("📊 Rata-rata per Text", f"{avg_tokens:.1f}")
        with col2:
            total_tokens = data['token_count'].sum()
            st.metric("📈 Total Tokens", f"{total_tokens:,}")

        # Show comparison
        st.markdown("**Perbandingan Text :**")
        comparison_df = data[['normalized', 'tokenized', 'token_count']].head(10)
        st.dataframe(comparison_df, use_container_width=True)

    # Tab 4: Stemmed
    with tab4:
        st.subheader("Step 4: Stemmed Text")
        st.markdown("**Operation:** Remove prefixes and suffixes from words")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before Stemming**")
            st.dataframe(
                data[['tokenized']].head(10),
                use_container_width=True,
                height=400
            )
        with col2:
            st.markdown("**After Stemming**")
            st.dataframe(
                data[['stemmed']].head(10),
                use_container_width=True,
                height=400
            )

    # Tab 5: Stopwords Removed (Final)
    with tab5:
        st.subheader("Step 5: Final Text (Stopwords Removed)")
        st.markdown("**Operation:** Remove common words with little semantic meaning")

        # Calculate word reduction
        data['words_with_stopwords'] = data['stemmed'].str.split().str.len()
        data['words_without_stopwords'] = data['stopword'].str.split().str.len()
        data['words_removed'] = data['words_with_stopwords'] - data['words_without_stopwords']

        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_before = data['words_with_stopwords'].sum()
            st.metric("📊 Words Before", f"{total_before:,}")
        with col2:
            total_after = data['words_without_stopwords'].sum()
            st.metric("📊 Words After", f"{total_after:,}")
        with col3:
            reduction_pct = ((total_before - total_after) / total_before * 100)
            st.metric("📉 Reduction", f"{reduction_pct:.1f}%")

        # Show comparison
        st.markdown("**Final Results:**")
        comparison_df = data[[
            'stemmed',
            'stopword',
            'words_removed'
        ]].head(10)
        comparison_df.columns = ['With Stopwords', 'Without Stopwords', 'Words Removed']
        st.dataframe(comparison_df, use_container_width=True)

        # Top removed words
        with st.expander("🔍 Most Removed Stopwords"):
            all_removed = []
            for idx, row in data.iterrows():
                before_words = set(str(row['stemmed']).split())
                after_words = set(str(row['stopword']).split())
                removed = before_words - after_words
                all_removed.extend(list(removed))
                
            if all_removed:
                from collections import Counter
                # Count the frequency of each removed word (stopwords)
                top_removed = Counter(all_removed).most_common(20)
                removed_df = pd.DataFrame.from_records(top_removed, columns=['Word', 'Frequency'])
                st.bar_chart(removed_df.set_index('Word'))


    # Tab 6: Overview
    with tab6:
        st.subheader("📈 Complete Processing Overview")

        # Summary statistics
        st.markdown("### 📊 Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📄 Total Texts", f"{len(data):,}")
        with col2:
            avg_original = data['full_text'].str.len().mean()
            st.metric("📏 Avg Original Length", f"{avg_original:.0f} chars")
        with col3:
            avg_final = data['stopword'].str.len().mean()
            st.metric("📏 Avg Final Length", f"{avg_final:.0f} chars")
        with col4:
            size_reduction = (1 - avg_final / avg_original) * 100
            st.metric("📉 Size Reduction", f"{size_reduction:.1f}%")

        # Processing pipeline visualization
        st.markdown("### 🔄 Processing Pipeline Impact")

        pipeline_stats = pd.DataFrame({
            'Stage': ['Original', 'Cleaned', 'Normalized', 'Tokenized', 'Stemmed', 'Final'],
            'Avg Length (chars)': [
                data['full_text'].str.len().mean(),
                data['cleaned'].str.len().mean(),
                data['normalized'].str.len().mean(),
                data['tokenized'].str.len().mean(),
                data['stemmed'].str.len().mean(),
                data['stopword'].str.len().mean()
            ]
        })

        st.line_chart(pipeline_stats.set_index('Stage'))

        # Sample comparison
        st.markdown("### 🔬 Sample Text Transformation")
        sample_idx = st.selectbox("Select sample to view:", range(min(20, len(data))))

        if sample_idx < len(data):
            sample = data.iloc[sample_idx]

            stages = {
                "1️⃣ Original": sample['full_text'],
                "2️⃣ Cleaned": sample['cleaned'],
                "3️⃣ Normalized": sample['normalized'],
                "4️⃣ Tokenized": sample['tokenized'],
                "5️⃣ Stemmed": sample['stemmed'],
                "6️⃣ Final": sample['stopword']
            }

            for stage, text in stages.items():
                st.markdown(f"**{stage}**")
                st.info(text)

        # Download section
        st.markdown("### 📥 Download Processed Data")
        col1, col2, col3 = st.columns(3)

        with col1:
            # Download all steps
            csv_full = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download All Steps",
                data=csv_full,
                file_name="preprocessed_full.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            # Download final only
            csv_final = data[['full_text', 'stopword']].to_csv(index=False).encode('utf-8')
            csv_final = csv_final.replace(b'stopword', b'preprocessed_text')  # Rename column
            st.download_button(
                label="✨ Download Final Text",
                data=csv_final,
                file_name="preprocessed_final.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col3:
            # Download with original
            csv_comparison = data[['full_text', 'cleaned', 'normalized', 'stemmed', 'stopword']].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="🔄 Download Comparison",
                data=csv_comparison,
                file_name="preprocessed_comparison.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Reset button
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Process New File", use_container_width=True, type="secondary"):
            # Clear session state
            st.session_state['uploaded_file'] = None
            st.session_state['processed_data'] = None
            st.rerun()

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
