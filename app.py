import os
# --- CRITICAL SERVER BYPASS FLAGS ---
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"

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

# --- UI DESIGN (Modern Dark Theme) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #1a1a1a;
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        background-color: #1a1a1a;
        padding: 20px;
        text-align: center;
        border-bottom: 2px solid #333;
        margin-bottom: 20px;
    }
    /* Button Styling */
    div.stButton > button {
        border-radius: 10px;
        height: 3.5em;
        font-weight: bold;
        width: 100%;
        transition: 0.3s;
        color: white;
        background-color: #1f538d;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #2980b9;
        border: 1px solid #ffffff;
    }
    /* File Uploader styling */
    .stFileUploader {
        background-color: #262626;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_yolo_model():
    # Downloads the yolov8n.pt model automatically on first run
    return YOLO("yolov8n.pt")

model = load_yolo_model()

# --- HEADER ---
st.markdown('<div class="header"><h1>YOLOv8 Vision Suite - Pro</h1></div>', unsafe_allow_html=True)

# --- NAVIGATION SYSTEM ---
if "mode" not in st.session_state:
    st.session_state.mode = "home"

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📁 Select Video"):
        st.session_state.mode = "video"
with col2:
    if st.button("🖼️ Select Image"):
        st.session_state.mode = "image"
with col3:
    if st.button("▶ Start Webcam"):
        st.session_state.mode = "webcam"
with col4:
    if st.button("⏹ Stop Feed"):
        st.session_state.mode = "home"

st.divider()

# --- APP LOGIC ---

# 1. VIDEO MODE
if st.session_state.mode == "video":
    st.subheader("🎥 Video Object Detection")
    uploaded_video = st.file_uploader("Upload a video file...", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        vf = cv2.VideoCapture(tfile.name)
        
        st_frame = st.empty()
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break
            
            # Predict and Plot
            results = model.predict(frame, conf=0.4, verbose=False)
            annotated_frame = results[0].plot()
            
            # Display frame
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
        vf.release()

# 2. IMAGE MODE
elif st.session_state.mode == "image":
    st.subheader("📸 Image Object Detection")
    uploaded_img = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "bmp", "webp"])
    
    if uploaded_img:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # Predict
        results = model.predict(img, conf=0.4, verbose=False)
        annotated_img = results[0].plot()
        
        st.image(annotated_img, channels="BGR", caption="Analysis Complete", use_container_width=True)

# 3. WEBCAM MODE (Back Camera Optimized)
elif st.session_state.mode == "webcam":
    st.subheader("📱 Real-Time Tracking (Back Camera)")
    
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        
        # ByteTrack logic as per your original desktop code
        results = model.track(img, persist=True, tracker="bytetrack.yaml", conf=0.4, verbose=False)
        annotated_frame = results[0].plot()
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    # STUN configuration for mobile network bypass
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="vision-pro-tracker",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "facingMode": "environment", # This forces the REAR camera on mobile
                "width": {"ideal": 1280},
                "height": {"ideal": 720}
            },
            "audio": False
        },
        async_processing=True,
    )

# 4. HOME MODE
else:
    st.info("System Ready. Please select a source from the Control Panel above to begin tracking.")
    st.markdown("""
    ### Features:
    * **Tracking:** Uses ByteTrack for persistent object IDs.
    * **Mobile Ready:** Automatically triggers the environment (back) camera.
    * **Versatile:** Supports real-time feeds, video files, and static images.
    """)
