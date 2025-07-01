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
YOUTUBE_API_KEY = "AIzaSyCqFyrK_QRFl1llBZ5TABF8N1ImFBQgNj4"

# Configure Streamlit
st.set_page_config(page_title="📊 YouTube Content Strategist", layout="wide")
st.title("🎥 YT Brain: The AI Toolkit for YouTube Success")

def seconds_to_minutes(seconds):
    return seconds / 60 if seconds > 0 else 0

def calculate_engagement(row):
    views = row['views'] if row['views'] > 0 else 1
    return (2 * row['likes'] + row['comments']) / views

import streamlit as st
from streamlit.components.v1 import html
import json

def add_music():
    st.sidebar.header("🎵 Music Visualizer")
    
    # YouTube URL input with example placeholder
    youtube_url = st.sidebar.text_input(
        "Enter YouTube Music URL:",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste a YouTube music URL to enable visualizations"
    )
    
    if youtube_url:
        try:
            video_id = youtube_url.split("v=")[1].split("&")[0]
            
            # Responsive music player
            st.sidebar.markdown(f"""
            <div style="border-radius: 10px; overflow: hidden; margin-bottom: 20px;">
                <iframe width="100%" height="200" 
                src="https://www.youtube.com/embed/{video_id}?autoplay=1&mute=0&enablejsapi=1" 
                frameborder="0" 
                style="border-radius: 10px;"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen></iframe>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualization selector
            viz_theme = st.sidebar.selectbox(
                "Visualization Theme",
                ["Particle Wave", "Audio Spectrum", "Energy Rings"],
                index=0
            )
            
            # Dynamic color picker
            viz_color = st.sidebar.color_picker(
                "Visualization Color", 
                "#FF4B4B" if "upbeat" in youtube_url else "#00FFAA"
            )
            
            # Generate the visualization
            show_visualization(video_id, viz_theme, viz_color)
            
        except IndexError:
            st.sidebar.error("Invalid YouTube URL format")

def show_visualization(video_id, theme, color):
    """Renders interactive music visualization"""
    hex_color = color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    html(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <style>
            #visualizer-container {{
                width: 100%;
                height: 400px;
                border-radius: 15px;
                background: linear-gradient(135deg, #1e1e2f 0%, #2d2d42 100%);
                margin-top: 20px;
                overflow: hidden;
            }}
            canvas {{ 
                display: block; 
            }}
        </style>
    </head>
    <body>
        <div id="visualizer-container">
            <canvas id="visualizer"></canvas>
        </div>
        
        <script>
            // Initialize Three.js scene
            const container = document.getElementById('visualizer-container');
            const canvas = document.getElementById('visualizer');
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, container.clientWidth/container.clientHeight, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({{ canvas: canvas, antialias: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            
            // Audio analysis setup
            let audioContext, analyser, dataArray;
            const fftSize = 256;
            
            // Create visualization elements based on theme
            {get_theme_js(theme, rgb)}
            
            // Animation loop
            function animate() {{
                requestAnimationFrame(animate);
                {get_animation_js(theme)}
                renderer.render(scene, camera);
            }}
            
            // Connect to YouTube player
            function connectAudio() {{
                try {{
                    const video = document.querySelector('iframe').contentWindow.document.getElementsByTagName('video')[0];
                    if (!video) return setTimeout(connectAudio, 500);
                    
                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const source = audioContext.createMediaElementSource(video);
                    analyser = audioContext.createAnalyser();
                    analyser.fftSize = fftSize;
                    source.connect(analyser);
                    dataArray = new Uint8Array(analyser.frequencyBinCount);
                    
                    animate();
                }} catch (e) {{
                    setTimeout(connectAudio, 500);
                }}
            }}
            
            // Start when ready
            setTimeout(connectAudio, 1000);
        </script>
    </body>
    </html>
    """, height=420)

def get_theme_js(theme, rgb):
    """Returns JS code for different visualization themes"""
    if theme == "Particle Wave":
        return f"""
        // Particle Wave Theme
        const particles = new THREE.BufferGeometry();
        const particleCount = 300;
        const positions = new Float32Array(particleCount * 3);
        
        for (let i = 0; i < particleCount; i++) {{
            positions[i * 3] = (i / particleCount - 0.5) * 10;
            positions[i * 3 + 1] = 0;
            positions[i * 3 + 2] = 0;
        }}
        
        particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const particleMaterial = new THREE.PointsMaterial({{
            size: 0.2,
            color: new THREE.Color({rgb[0]/255}, {rgb[1]/255}, {rgb[2]/255}),
            transparent: true,
            opacity: 0.8
        }});
        
        const particleSystem = new THREE.Points(particles, particleMaterial);
        scene.add(particleSystem);
        camera.position.z = 5;
        """
    elif theme == "Audio Spectrum":
        return f"""
        // Audio Spectrum Theme
        const bars = [];
        const barCount = 64;
        const barWidth = 0.1;
        const barSpacing = 0.05;
        
        for (let i = 0; i < barCount; i++) {{
            const geometry = new THREE.BoxGeometry(barWidth, 0.1, 0.1);
            const material = new THREE.MeshBasicMaterial({{
                color: new THREE.Color({rgb[0]/255}, {rgb[1]/255}, {rgb[2]/255}),
                transparent: true,
                opacity: 0.8
            }});
            
            const bar = new THREE.Mesh(geometry, material);
            bar.position.x = (i - barCount/2) * (barWidth + barSpacing);
            bar.position.y = -2;
            scene.add(bar);
            bars.push(bar);
        }}
        
        camera.position.z = 5;
        """
    else:  # Energy Rings
        return f"""
        // Energy Rings Theme
        const rings = [];
        const ringCount = 5;
        const ringSegments = 32;
        
        for (let i = 0; i < ringCount; i++) {{
            const radius = 1 + i * 0.5;
            const geometry = new THREE.BufferGeometry();
            const vertices = [];
            
            for (let j = 0; j <= ringSegments; j++) {{
                const theta = (j / ringSegments) * Math.PI * 2;
                vertices.push(Math.cos(theta) * radius, Math.sin(theta) * radius, 0);
            }}
            
            geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertices), 3));
            const material = new THREE.LineBasicMaterial({{
                color: new THREE.Color({rgb[0]/255}, {rgb[1]/255}, {rgb[2]/255}),
                transparent: true,
                opacity: 0.6 - (i * 0.1)
            }});
            
            const ring = new THREE.Line(geometry, material);
            scene.add(ring);
            rings.push(ring);
        }}
        
        camera.position.z = 5;
        """

def get_animation_js(theme):
    """Returns animation logic for each theme"""
    if theme == "Particle Wave":
        return """
        analyser.getByteFrequencyData(dataArray);
        const positions = particles.attributes.position.array;
        
        for (let i = 0; i < particleCount; i++) {
            const freqIndex = Math.floor(i / particleCount * (fftSize/2));
            positions[i * 3 + 1] = dataArray[freqIndex] / 50;
        }
        
        particles.attributes.position.needsUpdate = true;
        """
    elif theme == "Audio Spectrum":
        return """
        analyser.getByteFrequencyData(dataArray);
        
        for (let i = 0; i < bars.length; i++) {
            const freqIndex = Math.floor(i / bars.length * (fftSize/2));
            bars[i].scale.y = 1 + dataArray[freqIndex] / 50;
            bars[i].material.opacity = 0.3 + (dataArray[freqIndex] / 255) * 0.7;
        }
        """
    else:  # Energy Rings
        return """
        analyser.getByteFrequencyData(dataArray);
        
        for (let i = 0; i < rings.length; i++) {
            const scale = 1 + (dataArray[i * 10] / 255) * 0.5;
            rings[i].scale.set(scale, scale, scale);
            
            if (i % 2 === 0) {
                rings[i].rotation.z += 0.005;
            } else {
                rings[i].rotation.z -= 0.005;
            }
        }
        """




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

# Main App
def main():
    add_music()
    st.sidebar.header("Settings")
    topic = st.sidebar.text_input("Enter YouTube Topic", "Example - Deep Learning")
    max_results = st.sidebar.slider("Number of Videos to Analyze", 20, 100, 50)

    if st.sidebar.button("🚀 Analyze"):
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
                    plt.xlabel("Duration (seconds)")
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
