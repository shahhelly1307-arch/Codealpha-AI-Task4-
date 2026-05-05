import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# --- MODERN DARK THEME ---
st.set_page_config(page_title="YOLOv8 Vision Suite - Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #1a1a1a; color: white; }
    .header { text-align: center; border-bottom: 2px solid #333; padding: 10px; }
    div.stButton > button { width: 100%; border-radius: 10px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()

st.markdown('<div class="header"><h1>YOLOv8 Vision Suite - Pro</h1></div>', unsafe_allow_html=True)

# --- BACK CAMERA LOGIC ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # Performance tracking
    results = model.track(img, persist=True, tracker="bytetrack.yaml", verbose=False)
    annotated_frame = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# STUN servers help your phone connect to the cloud
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- THE INTERFACE ---
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Controls")
    # Setting facingMode to "environment" forces the BACK camera
    start = st.button("▶ Start Tracker")
    stop = st.button("⏹ Stop Feed")

with col2:
    if start:
        webrtc_streamer(
            key="tracker",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": {"facingMode": "environment"}, # THIS USES BACK CAMERA
                "audio": False
            },
            async_processing=True,
        )
    else:
        st.info("Click 'Start Tracker' to begin detection using your back camera.")
          
