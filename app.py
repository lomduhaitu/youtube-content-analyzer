import nltk
nltk.download('punkt', download_dir='/tmp')
nltk.download('wordnet', download_dir='/tmp')
nltk.download('stopwords', download_dir='/tmp')
nltk.download('vader_lexicon', download_dir='/tmp')
nltk.data.path.append('/tmp')


# app.py
import streamlit as st
import pandas as pd
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keybert import KeyBERT
from collections import Counter
import nltk
from youtube_data_handler import fetch_all_data

st.set_page_config(page_title="📊 YouTube Content Recommender", layout="wide")
st.title("🎥 YouTube Content Recommender")

topic = st.text_input("Enter a YouTube topic:", value="machine learning")

if st.button("🚀 Analyze"):
    with st.spinner("Fetching videos, analyzing comments..."):
        video_df, comments = fetch_all_data(topic)

    st.success("✅ Data fetched and processed!")

    def preprocess_comment(comment):
        comment = re.sub(r"http\S+", "", comment)
        comment = re.sub(r"[^a-zA-Z\s]", "", comment)
        words = comment.lower().split()
        words = [w for w in words if w not in stopwords.words("english")]
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w) for w in words]
        return " ".join(words)

    clean_comments = [preprocess_comment(c) for c in comments]

    sia = SentimentIntensityAnalyzer()
    def get_sentiment(comment):
        score = sia.polarity_scores(comment)
        if score['compound'] >= 0.05:
            return "Positive"
        elif score['compound'] <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    sentiments = [get_sentiment(c) for c in clean_comments]
    sentiment_counts = Counter(sentiments)

    kw_model = KeyBERT()
    keywords = [kw_model.extract_keywords(c, keyphrase_ngram_range=(1, 2), stop_words='english') for c in clean_comments]
    flat_keywords = [kw[0][0] for kw in keywords if kw]
    keyword_counts = Counter(flat_keywords)

    year = pd.Timestamp.now().year
    title_ideas = [f"🔥 {kw.title()} Trends in {year}" for kw, _ in keyword_counts.most_common(5)]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Sentiment Summary")
        st.metric("👍 Positive", sentiment_counts["Positive"])
        st.metric("😐 Neutral", sentiment_counts["Neutral"])
        st.metric("👎 Negative", sentiment_counts["Negative"])
    with col2:
        st.subheader("🔍 Top Keywords")
        for word, count in keyword_counts.most_common(5):
            st.markdown(f"- **{word}** ({count} mentions)")

    st.subheader("🎯 Suggested Video Titles")
    for title in title_ideas:
        st.markdown(f"- {title}")

    st.markdown("---")
    st.caption("💡 Built using YouTube API, NLP, and Sentiment Analysis")

# Set up Gemini API
genai.configure(api_key=st.secrets["gemini"]["api_key"])

    def generate_ai_recommendations(topic, top_keywords, sentiment_counts):
            """
            Calls Gemini API and returns AI-generated content strategy.
            """
            # Prepare enhanced prompt
            prompt = f"""
        You are a YouTube content strategist.
        
        A creator wants to make videos on the topic: "{topic}".
        
        Here is the data:
        - Frequently mentioned keywords from viewer comments: {top_keywords}
        - Audience sentiment analysis: {sentiment_counts}
        
        Based on this, please:
        1. Suggest 5 viral video title ideas
        2. Recommend an ideal video duration range (in minutes)
        3. Provide 5 relevant hashtags
        
        Be creative but data-informed. Keep suggestions short and engaging.
        """
        
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                return f"❌ Error generating recommendations: {e}"

# After keyword_counts, sentiment_counts are calculated...
st.subheader("🧠 AI-Powered Content Strategy")

ai_response = generate_ai_recommendations(topic, keyword_counts.most_common(5), sentiment_counts)
st.markdown(ai_response)


    
