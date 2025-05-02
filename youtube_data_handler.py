# youtube_data_handler.py
import pandas as pd

API_KEY = st.secrets["youtube_api_key"]

def fetch_all_data(topic):
    video_df = pd.read_csv("processed_metadata.csv")
    df = pd.read_csv("comments_flat.csv")
    comments = df["comment_text"].fillna("").astype(str).tolist()
    return video_df, comments
