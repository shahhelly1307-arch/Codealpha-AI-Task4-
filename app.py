import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import tempfile

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="YOLOv8 Vision Suite - Pro", layout="wide")

# --- UI DESIGN ---
st.markdown("""
    <style>
    .stApp { background-color: #1a1a1a; color: white; }
    .header {
        padding: 20px;
        text-align: center;
        border-bottom: 2px solid #333;
        margin-bottom: 20px;
    }
    div.stButton > button {
        border-radius: 10px;
        height: 3.5em;
        font-weight: bold;
        width: 100%;
        background-color: #1f538d;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_yolo_model():
    # Downloads yolov8n.pt automatically
    return YOLO("yolov8n.pt")

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- HEADER ---
st.markdown('<div class="header"><h1>YOLOv8 Vision Suite - Pro</h1></div>', unsafe_allow_html=True)

# --- NAVIGATION ---
if "mode" not in st.session_state:
    st.session_state.mode = "home"

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

# --- APP LOGIC ---

if st.session_state.mode == "video":
    st.subheader("🎥 Video Object Detection")
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        vf = cv2.VideoCapture(tfile.name)
        
        st_frame = st.empty()
        stop_btn = st.button("Stop Processing")
        
        while vf.isOpened() and not stop_btn:
            ret, frame = vf.read()
            if not ret: break
            
            results = model.predict(frame, conf=0.4, verbose=False)
            annotated_frame = results[0].plot()
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
        vf.release()

elif st.session_state.mode == "image":
    st.subheader("📸 Image Object Detection")
    uploaded_img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    
    if uploaded_img:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        results = model.predict(img, conf=0.4, verbose=False)
        st.image(results[0].plot(), channels="BGR", use_container_width=True)

elif st.session_state.mode == "webcam":
    st.subheader("📱 Real-Time Tracking")
    
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        # Use persist=True for tracking IDs
        results = model.track(img, persist=True, conf=0.4, verbose=False)
        annotated_frame = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="yolo-tracker",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {"facingMode": "environment"},
            "audio": False
        },
        async_processing=True,
    )

else:
    st.info("Select a source above to begin.")
