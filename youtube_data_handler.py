import nltk
from nltk.corpus import stopwords
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
import re

# Ensure stopwords are available
nltk.download('stopwords', quiet=True)

# Set up your YouTube API key and build the YouTube API client
YOUTUBE_API_KEY = "AIzaSyCqFyrK_QRFl1llBZ5TABF8N1ImFBQgNj4"  # Add your API key here
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Cache to avoid fetching data repeatedly
@st.cache_data
def fetch_all_data(topic):
    # Fetch YouTube video metadata
    video_df = fetch_video_metadata(topic)
    
    # Fetch YouTube comments
    comments = fetch_comments(topic)

    return video_df, comments

def fetch_video_metadata(topic):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)

    # Searching for videos related to the topic
    request = youtube.search().list(
        q=topic,
        part="snippet",
        maxResults=10  # Adjust the number of results to fit your use case
    )

    response = request.execute()
    video_data = []

    # Extracting metadata from the response
    for item in response['items']:
        video_id = item['id']['videoId']
        
        # Fetching statistics for the video
        stats_request = youtube.videos().list(
            part="statistics,contentDetails,snippet",
            id=video_id
        )
        
        stats_response = stats_request.execute()
        stats = stats_response['items'][0]
        
        # Extracting additional details
        category_name = stats['snippet'].get('categoryId', 'N/A')  # Category info
        tags = stats['snippet'].get('tags', [])
        hashtags = extract_hashtags(item['snippet']['description'])
        
        # Extracting keywords from title and description
        title_keywords = extract_keywords(item['snippet']['title'])
        description_keywords = extract_keywords(item['snippet']['description'])

        # Duration of the video
        duration = stats['contentDetails']['duration']
        
        # Video URL
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        # Video metadata dictionary
        video_info = {
            "title": item["snippet"]["title"],
            "channel": item["snippet"]["channelTitle"],
            "published_at": item["snippet"]["publishedAt"],
            "category": category_name,
            "tags": tags,
            "hashtags": hashtags,
            "title_keywords": title_keywords,
            "description_keywords": description_keywords,
            "views": stats.get("statistics", {}).get("viewCount", "N/A"),
            "likes": stats.get("statistics", {}).get("likeCount", "N/A"),
            "comments": stats.get("statistics", {}).get("commentCount", "N/A"),
            "duration": duration,
            "video_url": video_url
        }
        video_data.append(video_info)

    # Converting to a DataFrame
    video_df = pd.DataFrame(video_data)

    return video_df

def fetch_comments(topic):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)

    # Searching for videos related to the topic
    request = youtube.search().list(
        q=topic,
        part="snippet",
        maxResults=10  # Adjust the number of results to fit your use case
    )

    response = request.execute()
    
    comments = []
    for item in response['items']:
        video_id = item['id']['videoId']
        
        # Fetching comments for each video
        comment_request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=20  # Adjust number of comments to fetch per video
        )
        
        comment_response = comment_request.execute()
        
        for comment_item in comment_response['items']:
            comment = comment_item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
    
    # Limit to 200 comments for speed
    comments = comments[:200]
    
    return comments

def extract_hashtags(description):
    """ Extract hashtags from video description """
    return re.findall(r"#\w+", description)

def extract_keywords(text):
    """ Simple function to extract keywords from text using NLTK stopwords """
    # Get the NLTK stopwords list for English
    stop_words = set(stopwords.words('english'))
    
    # Tokenize the text and remove common stopwords
    words = re.findall(r'\w+', text.lower())  # Simple word tokenization
    keywords = [word for word in words if word not in stop_words]
    
    return keywords
