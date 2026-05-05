import os
import sys
import subprocess

# --- AUTO-INSTALLER FOR CLOUD ENVIRONMENT ---
try:
    import cv2
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2

import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import tempfile

# --- PAGE CONFIG ---
st.set_page_config(page_title="YOLOv8 Vision Suite - Pro", layout="wide")

# --- CUSTOM THEME (Dark Mode) ---
st.markdown("""
    <style>
    .stApp { background-color: #1a1a1a; color: white; font-family: 'Segoe UI', sans-serif; }
    .header { text-align: center; border-bottom: 2px solid #333; padding: 15px; margin-bottom: 20px; }
    div.stButton > button { border-radius: 10px; height: 3.5em; font-weight: bold; width: 100%; color: white; }
    /* Navigation Button Hover Effect */
    div.stButton > button:hover { border: 1px solid #1f538d; background-color: #1a1a1a; }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()

st.markdown('<div class="header"><h1>YOLOv8 Vision Suite - Pro</h1></div>', unsafe_allow_html=True)

if "mode" not in st.session_state:
    st.session_state.mode = "home"

# --- CONTROL PANEL ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("📁 Select Video"): st.session_state.mode = "video"
with col2:
    if st.button("🖼️ Select Image"): st.session_state.mode = "image"
with col3:
    if st.button("▶ Start Webcam"): st.session_state.mode = "webcam"
with col4:
    if st.button("⏹ Stop Feed"): st.session_state.mode = "home"

st.divider()

# --- APP MODULES ---

if st.session_state.mode == "video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        vf = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret: break
            results = model.predict(frame, conf=0.4, verbose=False)
            st_frame.image(results[0].plot(), channels="BGR", use_container_width=True)
        vf.release()

elif st.session_state.mode == "image":
    uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        img = Image.open(uploaded_img)
        img_array = np.array(img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        results = model.predict(img_bgr, conf=0.4, verbose=False)
        st.image(results[0].plot(), channels="BGR", use_container_width=True)

elif st.session_state.mode == "webcam":
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        # ByteTrack logic as requested
        results = model.track(img, persist=True, tracker="bytetrack.yaml", verbose=False)
        return av.VideoFrame.from_ndarray(results[0].plot(), format="bgr24")

    webrtc_streamer(
        key="pro-tracker",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {"facingMode": "environment"}, # BACK CAMERA
            "audio": False
        },
        async_processing=True,
    )
else:
    st.info("System Ready. Select a mode from the Control Panel.")
