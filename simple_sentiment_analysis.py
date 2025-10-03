# ===============================
# SIMPLE SENTIMENT ANALYSIS APP
# ===============================

# 1. Import library
import streamlit as st
from textblob import TextBlob
import pandas as pd
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# 2. Fungsi untuk mengubah hasil sentiment ke DataFrame
def convert_to_df(sentiment):
    sentiment_dict = {
        'polarity': sentiment.polarity,
        'subjectivity': sentiment.subjectivity
    }
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['metric', 'value'])
    return sentiment_df


# 3. Fungsi untuk analisis token (kata per kata)
def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []

    for word in docx.split():
        res = analyzer.polarity_scores(word)['compound']
        if res > 0.1:
            pos_list.append((word, res))
        elif res <= -0.1:
            neg_list.append((word, res))
        else:
            neu_list.append(word)

    result = {
        'positives': pos_list,
        'negatives': neg_list,
        'neutral': neu_list
    }
    return result


# 4. Fungsi utama aplikasi Streamlit
def main():
    st.title("Sentiment Analysis NLP App")
    st.subheader("Streamlit Projects")

    # Menu
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")

        # Input form
        with st.form("nlpForm"):
            raw_text = st.text_area("Masukkan teks di sini:")
            submit_button = st.form_submit_button(label='Analyze')

        # Layout kolom
        col1, col2 = st.columns(2)

        if submit_button:
            with col1:
                st.info("Sentiment Result")

                # Analisis menggunakan TextBlob
                sentiment = TextBlob(raw_text).sentiment
                st.write(sentiment)

                # Emoji & hasil sentimen
                if sentiment.polarity > 0:
                    st.markdown("**Sentiment:** 👍 Positive 😄")
                elif sentiment.polarity < 0:
                    st.markdown("**Sentiment:** 👎 Negative 😠")
                else:
                    st.markdown("**Sentiment:** 😐 Neutral")

                # DataFrame hasil
                result_df = convert_to_df(sentiment)
                st.dataframe(result_df)

                # Visualisasi
                chart = alt.Chart(result_df).mark_bar().encode(
                    x='metric',
                    y='value',
                    color='metric'
                )
                st.altair_chart(chart, use_container_width=True)

            with col2:
                st.info("Token Sentiment")
                token_sentiments = analyze_token_sentiment(raw_text)
                st.write(token_sentiments)

    else:
        st.subheader("About")


# Run program
if __name__ == '__main__':
    main()