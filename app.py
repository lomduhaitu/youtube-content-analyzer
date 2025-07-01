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
import google.generativeai as genai
from datetime import datetime
from PIL import Image
import io
import base64

# Preload NLTK Dat
nltk.download(['punkt', 'wordnet', 'stopwords', 'vader_lexicon'], quiet=True)

# Configure APIs
genai.configure(api_key= "AIzaSyCiWJwcl8u0tyOLYucXnA6JWlUwMMgvqbs")
YOUTUBE_API_KEY = "AIzaSyCqFyrK_QRFl1llBZ5TABF8N1ImFBQgNj4"

# Configure Streamlit
st.set_page_config(page_title="📊 YouTube Content Strategist", layout="wide")
st.title("🎥 YT Brain: The AI Toolkit for YouTube Success")

def seconds_to_minutes(seconds):
    return seconds / 60 if seconds > 0 else 0

def calculate_engagement(row):
    views = row['views'] if row['views'] > 0 else 1
    return (2 * row['likes'] + row['comments']) / views

# Add to your app.py
def add_music():
    st.sidebar.header("🎵 Background Music")
    youtube_url = st.sidebar.text_input("Enter YouTube Music URL:")
    
    if youtube_url:
        st.sidebar.write("Now playing:")
        st.sidebar.markdown(f"""
        <iframe width="100%" height="200" src="https://www.youtube.com/embed/{youtube_url.split('v=')[1].split('&')[0]}?autoplay=1&mute=0" 
        frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
        allowfullscreen></iframe>
        """, unsafe_allow_html=True)


# --- Gemini Content Expert Chatbot ---
def content_creator_chatbot():
    st.sidebar.header("🤖 Content Creator Assistant")
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Gemini API setup (FREE tier)
    genai.configure(api_key="AIzaSyCiWJwcl8u0tyOLYucXnA6JWlUwMMgvqbs")  # Replace with your key
    
    # Configure the model to act as a content expert
    generation_config = {
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 2000,
    }
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    # System prompt to make it a content expert
    expert_prompt = """You are ContentGPT, an expert digital content creator with 10+ years of experience. 
    You specialize in YouTube content strategy, scripting, SEO optimization, and audience engagement. 
    Provide concise, actionable advice. Always ask clarifying questions before giving recommendations."""
    
    # Chat UI
    user_query = st.sidebar.text_area("Ask about content creation:", height=100)
    
    if st.sidebar.button("Get Expert Advice"):
        with st.spinner("Consulting ContentGPT..."):
            try:
                # Include chat history for context
                full_prompt = f"{expert_prompt}\n\nChat History:\n"
                for msg in st.session_state.chat_history[-4:]:  # Last 4 messages
                    full_prompt += f"{msg['role']}: {msg['content']}\n"
                
                full_prompt += f"User: {user_query}"
                
                response = model.generate_content(full_prompt)
                
                # Store conversation
                st.session_state.chat_history.append({"role": "User", "content": user_query})
                st.session_state.chat_history.append({"role": "ContentGPT", "content": response.text})
                
                # Display response
                st.sidebar.success("ContentGPT says:")
                st.sidebar.markdown(f"```\n{response.text}\n```")
                
                # Show full chat in main area
                st.subheader("💬 Content Creation Chat History")
                for msg in st.session_state.chat_history:
                    if msg['role'] == "User":
                        st.markdown(f"**You**: {msg['content']}")
                    else:
                        st.markdown(f"**ContentGPT**: {msg['content']}")
                        st.divider()
            
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")


# Initialize Gemini (call this once in your app)
def init_gemini():
    genai.configure(api_key="AIzaSyCiWJwcl8u0tyOLYucXnA6JWlUwMMgvqbs")  # Replace with your key
    return genai.GenerativeModel('gemini-1.5-flash')

# Thumbnail Generator Tab
def thumbnail_tab(model):
    st.header("🎨 Professional Thumbnail Studio")
    
    with st.expander("⚙️ Thumbnail Brief", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            video_title = st.text_input("Video Title*", placeholder="5 Secrets to Viral Videos")
            content_type = st.selectbox(
                "Content Type*",
                ["Tutorial", "Review", "Vlog", "Gaming", "Tech", "Fashion", "Other"],
                index=2
            )
        with col2:
            target_audience = st.text_input("Target Audience*", placeholder="Ages 18-35, Tech Enthusiasts")
            style_preference = st.multiselect(
                "Style Preferences",
                ["Minimalist", "Bold Text", "Face Close-up", "Product Focus", "Dark Theme", "Bright Colors"],
                default=["Bold Text"]
            )
    
    with st.expander("🖼️ Reference Image (Optional)"):
        uploaded_image = st.file_uploader("Upload image for inspiration", type=["jpg", "png", "jpeg"])
        if uploaded_image:
            st.image(uploaded_image, caption="Your Reference", width=300)

    if st.button("✨ Generate Thumbnail Concepts", type="primary"):
        if not video_title or not content_type or not target_audience:
            st.warning("Please fill required fields (*)")
            return
            
        with st.spinner("Generating 3 professional concepts..."):
            try:
                prompt = f"""As a YouTube thumbnail expert, create 3 distinct concepts for:
Title: {video_title}
Content: {content_type}
Audience: {target_audience}
Styles: {', '.join(style_preference) if style_preference else 'Any'}

For each concept provide:
1. 🎭 Visual Composition - Describe layout, colors, key elements
2. ✏️ Text Treatment - Recommended text/fonts/placement
3. 🎯 Psychological Hook - Why it grabs attention
4. 💡 Pro Tip - Technical execution advice

Format with clear headings for each concept."""
                
                if uploaded_image:
                    img = Image.open(uploaded_image)
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format="PNG")
                    response = model.generate_content(
                        [prompt, {"mime_type": "image/png", "data": base64.b64encode(img_bytes.getvalue()).decode()}]
                    )
                else:
                    text_model = genai.GenerativeModel('gemini-pro')
                    response = text_model.generate_content(prompt)
                
                st.success("✅ Generated 3 Professional Concepts")
                st.session_state.last_thumbnail_response = response.text
                
                # Display with nice formatting
                for i, concept in enumerate(response.text.split("Concept ")[1:4], 1):
                    with st.container(border=True):
                        st.subheader(f"Concept {i}")
                        st.markdown(concept)
                
                # Download button
                st.download_button(
                    label="📥 Download All Concepts",
                    data=response.text,
                    file_name=f"thumbnail_concepts_{video_title[:30]}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"🚨 Generation failed: {str(e)}")




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
    5. 10 recommended hashtags
    
    Keep recommendations data-driven and actionable and content must be in english."""
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI recommendation error: {str(e)}"

def main():
    # Initialize page config (must be first Streamlit command)
    st.set_page_config(page_title="Content Creator Suite", layout="wide")
    
    # Initialize AI models
    model = init_gemini()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎵 Music Visualizer", "📹 Content Assistant", "🎨 Thumbnail Studio"])
    
    with tab1:
        add_music()  # Your music visualization function
    
    with tab2:
        content_creator_chatbot()  # Your existing chatbot function
        
        # YouTube Analysis Section (moved inside Content Assistant tab)
        st.header("YouTube Content Analysis")
        with st.sidebar:
            st.header("Settings")
            topic = st.text_input("Enter YouTube Topic", "Example - Deep Learning")
            max_results = st.slider("Number of Videos to Analyze", 20, 100, 50)

        if st.button("🚀 Analyze"):
            with st.spinner("Fetching and processing YouTube data..."):
                try:
                    # Get data
                    video_df, comments_df = fetch_all_data(topic, max_results)
                    
                    # Preprocess data
                    video_df['duration_mins'] = video_df['duration'].apply(seconds_to_minutes)
                    video_df = video_df[video_df['duration_mins'] > 0]  # Filter out bad data

                    video_df['engagement'] = video_df.apply(calculate_engagement, axis=1)
                    video_df['published_hour'] = pd.to_datetime(video_df['published_at']).dt.hour
                    
                    st.write(f"Analyzing {len(comments_df)} comments from {len(comments_df['video_id'].unique())} videos")
                    
                    # Sentiment analysis
                    sia = SentimentIntensityAnalyzer()
                    comments_df['sentiment'] = comments_df['comment'].apply(
                        lambda x: sia.polarity_scores(x)['compound']
                    )
                    
                    # Keyword extraction
                    kw_model = KeyBERT()
                    keywords = kw_model.extract_keywords(' '.join(comments_df['comment']), 
                                       keyphrase_ngram_range=(1, 2), top_n=20)
                    
                    # Prepare analysis data
                    analysis_data = {
                        'avg_duration': video_df.nlargest(10, 'engagement')['duration_mins'].median(),
                        'best_hours': video_df.groupby('published_hour')['engagement'].mean().nlargest(3).index.tolist(),
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
                    top_videos = video_df.nlargest(5, 'engagement')[['title', 'channel', 'duration_mins', 'engagement']]
                    
                    # Display as clickable links
                    for idx, row in top_videos.iterrows():
                        st.markdown(f"▶️ [{row['title']}](https://youtube.com/watch?v={idx})")
                        st.caption(f"Channel: {row['channel']} | Duration: {row['duration_mins']} mins | Engagement: {row['engagement']:.2f}")
                    
                except HttpError as e:
                    st.error(f"YouTube API Error: {str(e)}")
                except Exception as e:
                    st.error(f"Analysis Failed: {str(e)}")
    
    with tab3:
        thumbnail_tab(model)  # Your thumbnail generator function

if __name__ == "__main__":
    main()
