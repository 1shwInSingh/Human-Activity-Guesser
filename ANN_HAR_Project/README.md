# Human Activity Recognition with ANN

This project uses the UCI HAR dataset and a TensorFlow/Keras ANN to classify activities.

## Setup

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

This creates:
- `har_model.keras`
- `har_scaler.joblib`
- `X_test_raw.csv`
- `y_test.csv`

## Run the frontend

```bash
streamlit run app.py
```

Then open the local Streamlit URL in your browser.
