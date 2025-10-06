import streamlit as st
from textblob import TextBlob
import pandas as pd
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator


# --- Helper: Convert TextBlob sentiment to DataFrame ---
def convert_to_df(sentiment):
    sentiment_dict = {
        'polarity': sentiment.polarity,
        'subjectivity': sentiment.subjectivity
    }
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['Metric', 'Value'])
    return sentiment_df


# --- Helper: Cached translation for repeated words ---
def translate_cached(words, source='en', target='id'):
    translator = GoogleTranslator(source=source, target=target)
    cache = {}
    translated = []
    for w in words:
        if w not in cache:
            try:
                cache[w] = translator.translate(w)
            except Exception:
                cache[w] = w  # fallback to original if translation fails
        translated.append(cache[w])
    return translated


# --- Function: Token-level analysis with back translation ---
def analyze_token_sentiment_with_translation(text):
    analyzer = SentimentIntensityAnalyzer()
    translator_en = GoogleTranslator(source='auto', target='en')

    # Translate Indonesian -> English for analysis
    try:
        english_text = translator_en.translate(text)
    except Exception:
        english_text = text  # fallback if translation fails

    tokens = english_text.split()
    unique_tokens = list(set(tokens))  # avoid redundant translations

    # Analyze each token
    data = []
    for token in unique_tokens:
        score = analyzer.polarity_scores(token)['compound']
        if score > 0.1:
            category = 'Positif'
        elif score < -0.1:
            category = 'Negatif'
        else:
            category = 'Netral'
        data.append({'token_en': token, 'Skor': score, 'Kategori': category})

    # Translate tokens back to Indonesian
    indo_tokens = translate_cached([d['token_en'] for d in data], source='en', target='id')
    for i, indo_token in enumerate(indo_tokens):
        data[i]['Kata (Indo)'] = indo_token

    df = pd.DataFrame(data)[['Kata (Indo)', 'Skor', 'Kategori']]
    df = df.sort_values(by='Skor', ascending=False).reset_index(drop=True)
    return df


# --- Function: Main Streamlit app ---
def main():
    st.title("Sentiment Analysis Bahasa Indonesia")
    st.subheader("Analisis sentimen teks Bahasa Indonesia menggunakan translator otomatis")

    menu = ["Beranda", "Tentang"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Beranda":
        with st.form("nlpForm"):
            raw_text = st.text_area("Masukkan teks Bahasa Indonesia di sini:")
            submit_button = st.form_submit_button(label='Analisis')

        if submit_button and raw_text.strip():
            st.info("Hasil")

            col1, col2 = st.columns(2)

            with col1:
                # Step 1: Translation
                translator_en = GoogleTranslator(source='auto', target='en')
                translated_text = translator_en.translate(raw_text)

                st.markdown("### Teks Asli")
                st.write(raw_text)
                st.markdown("### Terjemahan ke Bahasa Inggris")
                st.write(translated_text)

                # Step 2: TextBlob sentiment
                sentiment = TextBlob(translated_text).sentiment
                st.markdown("### Hasil Analisis TextBlob")
                st.write(sentiment)

                # Step 3: Interpret polarity
                if sentiment.polarity > 0:
                    st.success("Sentimen: **Positif 😄**")
                elif sentiment.polarity < 0:
                    st.error("Sentimen: **Negatif 😠**")
                else:
                    st.info("Sentimen: **Netral 😐**")

                # Step 4: DataFrame + Visualization
                result_df = convert_to_df(sentiment)
                st.dataframe(result_df)

                c = alt.Chart(result_df).mark_bar().encode(
                    x='Metric',
                    y='Value',
                    color='Metric',
                    tooltip=['Metric', 'Value']
                ).properties(title='Visualisasi Sentimen (TextBlob)')
                st.altair_chart(c, use_container_width=True)

            with col2:
                st.markdown("### Analisis Token (VADER + Terjemahan Balik)")
                token_df = analyze_token_sentiment_with_translation(raw_text)
                st.dataframe(token_df)

    else:
        st.subheader("Tentang Aplikasi")
        st.markdown("""
        Aplikasi ini menggunakan **metode translasi** untuk menganalisis sentimen teks Bahasa Indonesia:

        1. Teks Bahasa Indonesia diterjemahkan ke Bahasa Inggris menggunakan *GoogleTranslator*.
        2. Analisis sentimen dilakukan menggunakan:
           - **TextBlob** → menghitung *polarity* & *subjectivity*
           - **VADER** → menganalisis sentimen per token (kata)
        3. Token yang telah dianalisis diterjemahkan kembali ke Bahasa Indonesia untuk ditampilkan ke pengguna.
        4. Hasil akhirnya menunjukkan kata-kata positif, negatif, dan netral beserta skornya.
        """)


if __name__ == '__main__':
    main()
