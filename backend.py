from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import librosa
from PIL import Image
from io import BytesIO

app = FastAPI()

# Enable CORS (needed if frontend is hosted separately)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once
model = tf.keras.models.load_model('C:/Users/karti/OneDrive/Desktop/Heart/model.h5')
_, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = model.input_shape
TARGET_SHAPE = (IMG_WIDTH, IMG_HEIGHT)
SAMPLE_RATE = 16000
FIXED_DURATION = 15
FIXED_LENGTH = SAMPLE_RATE * FIXED_DURATION

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    y, _ = librosa.load(BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    y = np.pad(y, (0, max(0, FIXED_LENGTH - len(y))), mode='constant')[:FIXED_LENGTH]

    # Preprocess: Create mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)
    S_DB = (S_DB - S_DB.min()) / (S_DB.max() - S_DB.min())  # Normalize to [0,1]

    # Resize spectrogram for model input
    img = Image.fromarray((S_DB * 255).astype(np.uint8))
    img = img.resize(TARGET_SHAPE, Image.BICUBIC)
    resized = np.array(img).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(resized, axis=-1)  # (H, W, 1)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # (1, H, W, 1)

    # Model prediction
    pred = model.predict(input_tensor)[0][0]

    # Return prediction and spectrogram array as JSON
    return {
        "prediction": float(pred),
        "spectrogram": resized.tolist()  # Convert numpy array to list for JSON
    }
