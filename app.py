import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
from keybert import KeyBERT
from youtube_data_handler import fetch_all_data  # Your data loading function

# Preload NLTK Data (fix for Streamlit Cloud)
import nltk
nltk.download('punkt', download_dir='/tmp')
nltk.download('wordnet', download_dir='/tmp')
nltk.download('stopwords', download_dir='/tmp')
nltk.download('vader_lexicon', download_dir='/tmp')
nltk.data.path.append('/tmp')

# Configure Streamlit
st.set_page_config(page_title="📊 YouTube Content Recommender", layout="wide")
st.title("🎥 YouTube Content Recommender")

# Topic input
topic = st.text_input("Enter a YouTube topic:", value="machine learning")

# Main App Logic
def calculate_engagement_matrix(video_df):
    # Filter videos with at least 10,000 views
    video_df = video_df[video_df['views'] >= 10000]

    # Calculate engagement metrics
    video_df['like_to_view_ratio'] = video_df['likes'] / video_df['views']
    video_df['comment_to_view_ratio'] = video_df['comments'] / video_df['views']

    # Select relevant columns for engagement matrix
    engagement_df = video_df[['title', 'likes', 'views', 'comments', 'like_to_view_ratio', 'comment_to_view_ratio']]

    return engagement_df

def plot_engagement_matrix(engagement_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.scatterplot(x='like_to_view_ratio', y='comment_to_view_ratio', size='views', data=engagement_df, sizes=(40, 200), ax=ax)
    ax.set_title('Engagement Matrix (Likes vs Comments per View)', fontsize=14)
    ax.set_xlabel('Like-to-View Ratio')
    ax.set_ylabel('Comment-to-View Ratio')
    st.pyplot(fig)

def generate_ai_recommendations(topic, top_keywords, sentiment_counts, top_tags):
    prompt = f"""
    You are a YouTube content strategist.
    A creator wants to make videos on the topic: "{topic}".
    Here is the data:
    - Frequently mentioned keywords from viewer comments: {top_keywords}
    - Audience sentiment analysis: {sentiment_counts}
    - Popular tags across videos: {top_tags}

    Based on this, please:
    1. Suggest 5 viral video title ideas
    2. Recommend an ideal video duration range (in minutes)
    3. Provide 5 relevant hashtags
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error generating recommendations: {e}"

# Calculate Sentiment and Keywords
def analyze_comments(comments):
    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()

    def get_sentiment(comment):
        score = sia.polarity_scores(comment)
        if score['compound'] >= 0.05:
            return "Positive"
        elif score['compound'] <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    sentiments = [get_sentiment(c) for c in comments]
    sentiment_counts = Counter(sentiments)

    # Keyword extraction
    kw_model = KeyBERT()
    full_text = " ".join(comments)
    keywords = kw_model.extract_keywords(full_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
    flat_keywords = [kw[0] for kw in keywords]
    keyword_counts = Counter(flat_keywords)

    return sentiment_counts, keyword_counts

# Streamlit navigation panel
tab1, tab2 = st.tabs(["🧑‍💻 Analyze", "📊 Visualize"])

with tab1:
    # Data analysis
    if st.button("🚀 Analyze"):
        with st.spinner("Fetching videos and comments..."):
            video_df, comments = fetch_all_data(topic)

        st.success("✅ Data fetched and processed!")

        # Analyze Comments
        sentiment_counts, keyword_counts = analyze_comments(comments)

        # Display Sentiment Summary
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Sentiment Summary")
            st.metric("👍 Positive", sentiment_counts["Positive"])
            st.metric("😐 Neutral", sentiment_counts["Neutral"])
            st.metric("👎 Negative", sentiment_counts["Negative"])

        # Display AI-powered Recommendations
        st.subheader("🧠 AI-Powered Content Strategy")
        ai_response = generate_ai_recommendations(
            topic, keyword_counts.most_common(5), sentiment_counts, top_tags
        )
        st.markdown(ai_response)

with tab2:
    # Engagement Matrix Visualization
    if 'video_df' in locals():
        engagement_df = calculate_engagement_matrix(video_df)
        plot_engagement_matrix(engagement_df)
    else:
        st.warning("Please analyze first to generate engagement data.")

# Footer
st.markdown("---")
st.caption("💡 Built using YouTube API, Gemini LLM, NLP and KeyBERT")
