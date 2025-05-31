# 🫀 Heart Sound Classification using PCG Signals

This project uses machine learning to classify heart sounds from PCG (Phonocardiogram) recordings. It can help detect conditions like murmurs or abnormal heart rhythms.

---

## 📦 Features

 upload heart sounds
- Preprocess and convert to spectrograms
- Classify using a deep learning model
- User-friendly web UI (Streamlit)
- Real-time prediction API (FastAPI)

---



## 📊 Dataset

- **Source**: [PhysioNet 2016 Challenge](https://physionet.org/content/challenge-2016/)
- **Files**: `.wav` audio files + `.txt` labels
- **Classes**: `normal`, `abnormal`

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
Load the model in your project file 
```

### 3. Run the web app
```bash

streamlit run app.py
```

### 4. Run the API
```bash

uvicorn main:app --reload
```

---

## 📧 Contact

For questions, open a GitHub issue or email: [kartikguptaasg@gmail.com)
