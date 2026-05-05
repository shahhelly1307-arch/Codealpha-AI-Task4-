import os
# Force headless mode to prevent libGL/libgthread errors
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import tempfile

# --- PAGE CONFIG ---
st.set_page_config(page_title="YOLOv8 Vision Suite - Pro", layout="wide")

# --- UI DESIGN (Matching your CustomTkinter Theme) ---
st.markdown("""
    <style>
    .stApp { background-color: #1a1a1a; color: white; }
    .header { text-align: center; border-bottom: 2px solid #333; padding: 15px; margin-bottom: 20px; }
    div.stButton > button { border-radius: 10px; height: 3.5em; font-weight: bold; color: white; width: 100%; }
    /* Match your specific button colors */
    .stButton>button:first-child { background-color: #1f538d; } /* Video */
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()

# --- HEADER ---
st.markdown('<div class="header"><h1>YOLOv8 Vision Suite - Pro</h1></div>', unsafe_allow_html=True)

# --- NAVIGATION STATE ---
if "mode" not in st.session_state:
    st.session_state.mode = "home"

# --- CONTROL PANEL BUTTONS ---
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

# --- FUNCTIONALITY ---

# 1. VIDEO ANALYSIS
if st.session_state.mode == "video":
    st.subheader("🎥 Video Object Detection")
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        vf = cv2.VideoCapture(tfile.name)
        
        st_frame = st.empty()
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret: break
            
            # Predict and Plot
            results = model.predict(frame, conf=0.4, verbose=False)
            res_plotted = results[0].plot()
            
            # Convert to RGB for Streamlit
            st_frame.image(res_plotted, channels="BGR", use_container_width=True)
        vf.release()

# 2. IMAGE ANALYSIS
elif st.session_state.mode == "image":
    st.subheader("🖼️ Image Object Detection")
    uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_img:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        results = model.predict(img, conf=0.4, verbose=False)
        st.image(results[0].plot(), channels="BGR", caption="Detection Result", use_container_width=True)

# 3. WEBCAM (BACK CAMERA)
elif st.session_state.mode == "webcam":
    st.subheader("📱 Real-Time Back Camera Tracker")

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        # Persistent tracking with ByteTrack
        results = model.track(img, persist=True, tracker="bytetrack.yaml", conf=0.4, verbose=False)
        annotated_frame = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    webrtc_streamer(
        key="vision-pro",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {"facingMode": "environment"}, # FORCES BACK CAMERA
            "audio": False
        },
        async_processing=True,
    )

else:
    st.info("Select a mode above to begin your analysis.")

