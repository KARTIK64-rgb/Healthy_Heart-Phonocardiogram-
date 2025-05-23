import streamlit as st
import requests
import matplotlib.pyplot as plt
import librosa
import numpy as np
from PIL import Image
import io

# Constants for frontend display and audio handling
SAMPLE_RATE = 16000
FIXED_DURATION = 15
FIXED_LENGTH = SAMPLE_RATE * FIXED_DURATION
TARGET_SHAPE = (1922, 128)  # Make sure this matches the backend preprocessing

# UI
st.title("🩺 PCG Heart Sound Classification")
st.write("Upload a .wav file to classify as Healthy (0) or Unhealthy (1).")

# Helper: Visualize spectrogram from backend response (optional)
def display_spectrogram(image_array):
    plt.figure(figsize=(4, 3))
    plt.imshow(image_array, aspect='auto', origin='lower', cmap='viridis')
    plt.axis('off')
    st.pyplot(plt)

# Upload file
audio_file = st.file_uploader("🎵 Upload a heart sound (.wav)", type=["wav"])
if audio_file is not None:
    st.audio(audio_file, format='audio/wav')

    try:
        with st.spinner("🔄 Sending audio to backend for prediction..."):
            # Send the uploaded file to FastAPI backend
            files = {'file': (audio_file.name, audio_file, 'audio/wav')}
            response = requests.post("http://localhost:8000/predict/", files=files)

        # Handle response
        if response.ok:
            result = response.json()
            pred = result["prediction"]
            confidence = pred * 100

            # Display result
            st.write(f"🧠 Raw model output (sigmoid): {pred:.4f}")
            if pred > 0.4:
                st.error(f"🔴 *Unhealthy Heart (1)* — Confidence: {confidence:.2f}%")
            else:
                st.success(f"🟢 *Healthy Heart (0)* — Confidence: {100 - confidence:.2f}%")

            # Optionally: display the spectrogram from the backend if included
            if "spectrogram" in result:
               img_data = np.array(result["spectrogram"])
               display_spectrogram(img_data)

        else:
            st.error("❌ Backend error occurred.")
            st.text(response.text)

    except Exception as e:
        st.error("❌ Could not contact the backend.")
        st.exception(e)
