# app.py
import nltk
import plotly.express as px
import pandas as pd
import streamlit as st
from streamlit_extras.card import card
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.metric_cards import style_metric_cards
from googleapiclient.errors import HttpError
import google.generativeai as genai
from youtube_data_handler import fetch_all_data
from nltk.sentiment import SentimentIntensityAnalyzer
from keybert import KeyBERT

# Preload NLTK Data
nltk.download(['punkt', 'wordnet', 'stopwords', 'vader_lexicon'], quiet=True)

genai.configure(api_key= "AIzaSyCiWJwcl8u0tyOLYucXnA6JWlUwMMgvqbs")
YOUTUBE_API_KEY = "AIzaSyDYXAlo8fSiynRZil2fek_vJRvOCjrHJPA"

# Configure Streamlit
st.set_page_config(page_title="📊 YouTube Content Strategist", layout="wide")
st.title("🎥 AI-Powered YouTube Content Optimizer")

def seconds_to_minutes(seconds):
    return seconds / 60 if seconds > 0 else 0

def calculate_engagement(row):
    views = max(int(row.get('views', 1)), 1)
    likes = int(row.get('likes', 0))
    comments = int(row.get('comments', 0))
    return (2 * likes + comments) / views

def generate_ai_recommendations(topic, analysis_data):
    prompt = f"""As a YouTube strategy expert, analyze this data and provide recommendations:
    
    Topic: {topic}
    Performance Analysis:
    - Average Duration: {analysis_data['avg_duration']:.1f} mins
    - Best Posting Hours: {analysis_data['best_hours']}
    - Top Keywords: {analysis_data['top_keywords']}
    - Audience Sentiment: {analysis_data['sentiment']:.2f}
    
    Provide:
    1. 5 viral title ideas
    2. Ideal video length range
    3. Best posting times
    4. 3 content strategy tips
    5. 10 recommended hashtags
    
    Use markdown formatting with bold headings."""
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ AI recommendation error: {str(e)}"

def display_analysis(video_df, comments_df, analysis_data, topic, recommendations):
    # Page Header
    with stylable_container(
        key="header",
        css_styles="""
            {background: linear-gradient(90deg, #4B79A1 0%, #283E51 100%);
            border-radius: 0.5rem;
            padding: 1rem;
            color: white;}
        """
    ):
        st.markdown(f"<h1 style='text-align: center;'>📈 {topic} Analysis Report</h1>", 
                   unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    # Replace the problematic card component with this corrected version
   # Replace the card section with this
    with col1:
        with st.container():
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
                    border-radius: 10px;
                    padding: 1.5rem;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    color: white;
                ">
                <h3 style="margin:0;padding:0;">🔥 Engagement Score</h3>
                <div style="font-size: 2.5rem; font-weight: bold; margin: 1rem 0;">
                    {video_df['engagement'].mean():.2f}
                </div>
                <div style="font-size: 0.9rem;">
                    Higher is better ({len(video_df)} videos analyzed)
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
    metric_cols = [col2, col3, col4]
    metrics = [
        ("⏳ Optimal Duration", f"{analysis_data['avg_duration']:.1f} mins", "Median of top videos"),
        ("⏰ Best Time", f"{analysis_data['best_hours'][0]:02}:00", "Peak engagement hour"),
        ("😊 Sentiment", f"{analysis_data['sentiment']:.2f}", "-1 (Negative) to +1 (Positive)")
    ]
    
    for col, (label, value, help_text) in zip(metric_cols, metrics):
        with col:
            st.metric(label=label, value=value, help=help_text)
    
    style_metric_cards()
    
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["📊 Duration Analysis", "⏰ Timing Insights", "🗣️ Audience Feedback"])
    
    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            fig = px.histogram(video_df, x='duration_mins', nbins=20,
                             title="Video Duration Distribution",
                             labels={'duration_mins': 'Minutes'},
                             color_discrete_sequence=['#4B79A1'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            with st.expander("📌 Duration Guide", expanded=True):
                st.markdown("""
                - **Short (<5min):** Quick tutorials
                - **Medium (8-15min):** Product reviews
                - **Long (>20min):** Deep dives
                """)
            st.progress(analysis_data['avg_duration']/40, text="Optimality Score")
    
    with tab2:
        fig = px.bar(video_df.groupby('published_hour')['engagement'].mean().reset_index(),
                   x='published_hour', y='engagement', 
                   labels={'published_hour': 'Hour', 'engagement': 'Score'},
                   color='engagement', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns([2, 3])
        with col1:
            sentiment_dist = comments_df['sentiment'].apply(
                lambda x: 'Positive' if x > 0.1 else 'Negative' if x < -0.1 else 'Neutral'
            ).value_counts()
            fig = px.pie(sentiment_dist, names=sentiment_dist.index, 
                        values=sentiment_dist.values, hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            keywords_df = pd.DataFrame(analysis_data['top_keywords'], columns=['Keyword'])
            st.dataframe(keywords_df.style.highlight_max(color='#4B79A1'), 
                        height=300, hide_index=True)
    
    # AI Recommendations
    st.markdown("## 🧠 Strategy Recommendations")
    with stylable_container(
        key="recommendations",
        css_styles="""
            {background: #f8f9fa;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
        """
    ):
        st.markdown(recommendations, unsafe_allow_html=True)
    
    # Top Videos
    st.markdown("## 🏆 Top Performing Videos")
    top_videos = video_df.nlargest(3, 'engagement')
    cols = st.columns(3)
    for idx, (_, row) in enumerate(top_videos.iterrows()):
        with cols[idx]:
            st.video(row['video_url'])
            st.caption(f"""
                **{row['title']}**  
                👤 {row['channel']}  
                ⏱️ {row['duration_mins']:.1f}m ❤️ {row['likes']}k 💬 {row['comments']}
            """)

def main():
    st.sidebar.header("⚙️ Settings")
    topic = st.sidebar.text_input("Enter YouTube Topic", "AI Technology")
    max_results = st.sidebar.slider("Videos to Analyze", 20, 100, 50)

    if st.sidebar.button("🚀 Start Analysis"):
        with st.spinner("🔍 Analyzing YouTube content..."):
            try:
                video_df, comments_df = fetch_all_data(topic, max_results)
                
                if video_df.empty:
                    st.error("No videos found! Try a different topic.")
                    return
                
                # Data Processing
                video_df['duration_mins'] = video_df['duration'].apply(seconds_to_minutes)
                video_df = video_df[video_df['duration_mins'] > 0]
                video_df['engagement'] = video_df.apply(calculate_engagement, axis=1)
                video_df['published_hour'] = pd.to_datetime(video_df['published_at']).dt.hour
                
                # Sentiment Analysis
                sia = SentimentIntensityAnalyzer()
                comments_df['sentiment'] = comments_df['comment'].apply(
                    lambda x: sia.polarity_scores(str(x))['compound']
                )
                
                # Keyword Extraction
                kw_model = KeyBERT()
                keywords = kw_model.extract_keywords(
                    ' '.join(comments_df['comment'].astype(str)),
                    keyphrase_ngram_range=(1, 2), 
                    top_n=15
                )
                
                analysis_data = {
                    'avg_duration': video_df.nlargest(10, 'engagement')['duration_mins'].median(),
                    'best_hours': video_df.groupby('published_hour')['engagement'].mean().nlargest(3).index.tolist(),
                    'top_keywords': [kw[0] for kw in keywords if kw[1] > 0.3],
                    'sentiment': comments_df['sentiment'].mean()
                }
                
                recommendations = generate_ai_recommendations(topic, analysis_data)
                display_analysis(video_df, comments_df, analysis_data, topic, recommendations)
                
                with st.expander("📁 Raw Data"):
                    st.write("### Video Data", video_df)
                    st.write("### Comments Data", comments_df)

            except HttpError as e:
                st.error(f"YouTube API Error: {e}")
            except Exception as e:
                st.error(f"Analysis Failed: {str(e)}")

if __name__ == "__main__":
    main()
