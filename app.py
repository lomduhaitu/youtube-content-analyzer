# app.py
import nltk
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import isodate
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.generativeai as genai
from youtube_data_handler import fetch_all_data
from nltk.sentiment import SentimentIntensityAnalyzer
from keybert import KeyBERT
from collections import Counter

# Preload NLTK Dat
nltk.download(['punkt', 'wordnet', 'stopwords', 'vader_lexicon'], quiet=True)

# Configure APIs
genai.configure(api_key= "AIzaSyCiWJwcl8u0tyOLYucXnA6JWlUwMMgvqbs")
YOUTUBE_API_KEY = "AIzaSyB-ZRiUSq9GEfj9eJ0TIDDLa8YMCqVW0R0"

# Configure Streamlit
st.set_page_config(page_title="📊 YouTube Content Strategist", layout="wide")
st.title("🎥 AI-Powered YouTube Content Optimizer")

def parse_duration(duration):
    try:
        return isodate.parse_duration(duration).total_seconds() / 60
    except Exception as e:
        print(f"Failed to parse duration: {duration} → {e}")
        return 0

def calculate_engagement(row):
    views = row['views'] if row['views'] > 0 else 1
    return (2 * row['likes'] + row['comments']) / views

# AI Recommendation Generator
def generate_ai_recommendations(topic, analysis_data):
    prompt = f"""As a YouTube strategy expert, analyze this data and provide recommendations:
    
    Topic: {topic}
    Performance Analysis:
    - Average Duration of Top Videos: {analysis_data['avg_duration']:.1f} mins
    - Best Posting Hours: {analysis_data['best_hours']}
    - Top Keywords: {analysis_data['top_keywords']}
    - Audience Sentiment: {analysis_data['sentiment']}
    
    Provide:
    1. 5 viral title ideas
    2. Ideal video length range
    3. Best posting times
    4. 3 content strategy tips
    5. 5 recommended hashtags
    
    Keep recommendations data-driven and actionable."""
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI recommendation error: {str(e)}"

# Main App
def main():
    st.sidebar.header("Settings")
    topic = st.sidebar.text_input("Enter YouTube Topic", "Tech Reviews")
    max_results = st.sidebar.slider("Number of Videos to Analyze", 20, 100, 50)

    if st.sidebar.button("🚀 Analyze"):
        with st.spinner("Fetching and processing YouTube data..."):
            try:
                # Get data
                video_df, comments_df = fetch_all_data(topic, max_results)
                
                # Preprocess data
                video_df['duration_mins'] = video_df['duration'].apply(parse_duration)
                video_df = video_df[video_df['duration_mins'] > 0]  # Filter out bad data

                video_df['engagement'] = video_df.apply(calculate_engagement, axis=1)
                video_df['published_hour'] = pd.to_datetime(video_df['published_at']).dt.hour
                
                # Sentiment analysis
                sia = SentimentIntensityAnalyzer()
                comments_df['sentiment'] = comments_df['comment'].apply(
                    lambda x: sia.polarity_scores(x)['compound']
                )
                
                # Keyword extraction
                kw_model = KeyBERT()
                keywords = kw_model.extract_keywords(' '.join(comments_df['comment']), 
                                   keyphrase_ngram_range=(1, 2), top_n=10)
                
                # Prepare analysis data
                analysis_data = {
                    'avg_duration': video_df.nlargest(10, 'engagement')['duration_mins'].median(),
                    'best_hours': video_df.groupby('published_hour')['engagement'].idxmax().values[:3],
                    'top_keywords': [kw[0] for kw in keywords],
                    'sentiment': comments_df['sentiment'].mean()
                }
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                # Visualization Section
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📈 Performance Insights")
                    fig, ax = plt.subplots()
                    sns.histplot(video_df['duration_mins'], bins=15, kde=True, ax=ax)
                    plt.xlabel("Duration (minutes)")
                    plt.title("Video Duration Distribution")
                    st.pyplot(fig)
                    
                with col2:
                    fig, ax = plt.subplots()
                    hour_engagement = video_df.groupby('published_hour')['engagement'].mean()
                    sns.barplot(x=hour_engagement.index, y=hour_engagement.values, ax=ax)
                    plt.title("Engagement by Posting Hour")
                    plt.xlabel("Hour of Day")
                    st.pyplot(fig)
                
                # AI Recommendations
                st.subheader("🧠 AI-Powered Strategy Recommendations")
                recommendations = generate_ai_recommendations(topic, analysis_data)
                st.markdown(recommendations)
                
                # Reference Videos
                st.subheader("🎬 Top Performing Reference Videos")
                top_videos = video_df.nlargest(5, 'engagement')[['title', 'channel', 'duration_mins', 'engagement', 'video_url']]
                st.dataframe(top_videos)
                
                # Raw Data
                with st.expander("View Raw Data"):
                    st.write("Video Data:", video_df)
                    st.write("Comments Data:", comments_df)
                    
            except HttpError as e:
                st.error(f"YouTube API Error: {str(e)}")
            except Exception as e:
                st.error(f"Analysis Failed: {str(e)}")

if __name__ == "__main__":
    main() 
