import nltk
from nltk.corpus import stopwords
import pandas as pd
from googleapiclient.discovery import build
import re
import time
nltk.download('stopwords', quiet=True)

YOUTUBE_API_KEY = "AIzaSyCqFyrK_QRFl1llBZ5TABF8N1ImFBQgNj4"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

def extract_hashtags(description):
    return re.findall(r"#\w+", description)

def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    words = re.findall(r'\w+', text.lower())
    return [word for word in words if word not in stop_words]

def fetch_videos_by_order(topic, order_type):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(
        q=topic,
        part="snippet",
        maxResults=10,
        order=order_type,
        type="video"
    )
    response = request.execute()

    video_data = []
    video_ids = []

    for item in response['items']:
        video_id = item['id']['videoId']
        video_ids.append(video_id)

        stats_request = youtube.videos().list(
            part="statistics,contentDetails,snippet",
            id=video_id
        )
        stats_response = stats_request.execute()
        stats = stats_response['items'][0]

        video_info = {
            "title": stats["snippet"]["title"],
            "channel": stats["snippet"]["channelTitle"],
            "published_at": stats["snippet"]["publishedAt"],
            "category": stats['snippet'].get('categoryId', 'N/A'),
            "tags": stats['snippet'].get('tags', []),
            "hashtags": extract_hashtags(stats['snippet'].get('description', '')),
            "title_keywords": extract_keywords(stats['snippet'].get('title', '')),
            "description_keywords": extract_keywords(stats['snippet'].get('description', '')),
            "views": stats.get("statistics", {}).get("viewCount", "N/A"),
            "likes": stats.get("statistics", {}).get("likeCount", "N/A"),
            "comments": stats.get("statistics", {}).get("commentCount", "N/A"),
            "duration": stats['contentDetails']['duration'],
            "video_url": f"https://www.youtube.com/watch?v={video_id}"
        }
        video_data.append(video_info)

    return video_data, video_ids

def fetch_top_comments(video_ids):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
    comments = []

    for video_id in video_ids:
        try:
            comment_request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                order="relevance",
                maxResults=100
            )
            comment_response = comment_request.execute()

            top_comments = sorted(
                [
                    {
                        "video_id": video_id,
                        "comment": item['snippet']['topLevelComment']['snippet']['textDisplay'],
                        "likes": item['snippet']['topLevelComment']['snippet']['likeCount']
                    }
                    for item in comment_response.get('items', [])
                ],
                key=lambda x: x['likes'],
                reverse=True
            )[:10]  # take top 10 liked

            comments.extend(top_comments)
        except Exception:
            continue

        time.sleep(0.1)  # slight delay to avoid quota issues

    return comments

def fetch_all_data(topic):
    categories = {
        "recent": "date",
        "most_viewed": "viewCount",
        "trending": "relevance",
        "most_liked": "rating"
    }

    combined_video_data = []
    combined_video_ids = []
    categorized_comments = {}

    for category, order_type in categories.items():
        video_data, video_ids = fetch_videos_by_order(topic, order_type)
        combined_video_data.extend(video_data)
        combined_video_ids.extend(video_ids)
        categorized_comments[category] = fetch_top_comments(video_ids)

    video_df = pd.DataFrame(combined_video_data)
    comments_df = pd.DataFrame(
        [(cat, com['video_id'], com['comment'], com['likes']) for cat, com_list in categorized_comments.items() for com in com_list],
        columns=["category", "video_id", "comment", "likes"]
    )

    return video_df, comments_df

# Example call:
# video_df, comments_df = fetch_all_data("Artificial Intelligence")
