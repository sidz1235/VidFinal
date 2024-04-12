import streamlit as st
import pandas as pd
from video_analysis import compute

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

st.title("Video Analysis")

uploaded_file = st.file_uploader("Upload video file", type=["mp4","mp3"])

if uploaded_file is not None:
       
    with st.spinner("Generating analysis..."):
        video_info = compute(uploaded_file)
    
    if video_info is not None:
        st.success("Video processed successfully!")

        flat_video_info = flatten_dict(video_info)

        df = pd.DataFrame.from_dict(flat_video_info, orient='index', columns=['Value'])

        st.table(df)
    else:
        st.error("Please upload a file.")
