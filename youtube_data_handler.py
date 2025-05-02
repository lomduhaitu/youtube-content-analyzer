# youtube_data_handler.py
import pandas as pd

@st.cache_data
def fetch_all_data(topic):
    video_df = pd.read_csv("processed_metadata.csv")
    df = pd.read_csv("comments_flat.csv")
    comments = df["comment_text"].fillna("").astype(str).tolist()
    comments = comments[:200]  # Limit for speed

    return video_df, comments
