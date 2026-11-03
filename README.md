# 🏃 Human Activity Guesser (HAR with ANN)

A machine learning web application that classifies human physical activities (Walking, Walking Up, Walking Down, Sitting, Standing, Laying) from smartphone sensor data using an Artificial Neural Network (ANN) trained on the **UCI Human Activity Recognition (HAR) Dataset**.

- **Dataset:** UCI HAR Dataset (561 features per sensor window)
- **Model:** 3-layer Deep Neural Network (Input 561 &rarr; Dense 256 &rarr; Dense 128 &rarr; Dense 6 Softmax)
- **Test Accuracy:** **95.4%**
- **Deployment:** Vercel-ready serverless Python runtime with lightweight pure NumPy inference (no heavy TensorFlow runtime required in production).

---

## 🚀 Live Deployment on Vercel

This repository is pre-configured for **1-click deployment on Vercel**:

1. Import this repository into [Vercel](https://vercel.com).
2. Leave all build and output settings as default.
3. Click **Deploy**.

Vercel automatically detects `api/index.py` and `app.py`, exposes the top-level `app` / `handler` WSGI application, and serves the interactive dashboard.

---

## 💻 Local Development

### 1. Run the Web App (Flask + NumPy)
```bash
pip install -r requirements.txt
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

### 2. Run the Streamlit Version (Optional)
If you prefer running the original Streamlit interface locally:
```bash
pip install streamlit tensorflow
streamlit run ANN_HAR_Project/streamlit_app.py
```

---

## 🧠 Training & Model Pipeline

To re-train the model or regenerate artifacts:
```bash
pip install -r ANN_HAR_Project/requirements-train.txt
python ANN_HAR_Project/train_model.py
```
This trains the Keras ANN model and generates:
- `har_model.keras`
- `har_scaler.joblib`
- `X_test_raw.csv` & `y_test.csv`
- `model_weights.npz` & `test_samples.npz` (for lightweight serverless inference)

