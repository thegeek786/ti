import logging
from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
from model.minimal_model import GCN_TCN_CapsNet

# Initialize Flask app
app = Flask(__name__)

# Logging config
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global prediction store
latest_prediction = {"predicted_activity": "Waiting for data..."}

# Load model
logger.info("Loading model...")
model = GCN_TCN_CapsNet(input_dim=3, num_classes=6, num_nodes=128)
model.load_state_dict(torch.load('model/saved_model_vvip.pth', map_location=torch.device('cpu')))
model.eval()
logger.info("Model loaded successfully.")

# Threshold to classify as Stationary
ACCELERATION_THRESHOLD = 0.2

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/data", methods=["POST"])
def receive_data():
    try:
        data = request.get_json(force=True)
        logger.debug("=== 📥 RAW DATA RECEIVED ===")
        logger.debug(data)

        if not data:
            logger.warning("No data received")
            return jsonify({"error": "No data received"}), 400

        raw_series = data.get("payload", [])
        logger.debug(f"📊 Payload entries: {len(raw_series)}")
        acc_count = sum(1 for entry in raw_series if 'accelerometer' in entry['name'])
        logger.debug(f"📈 Accelerometer readings: {acc_count}")

        # Preprocess
        input_tensor, accelerometer_data = preprocess_input(raw_series)
        logger.debug(f"✅ Preprocessed tensor shape: {input_tensor.shape}")

        # Check motion threshold
        avg_magnitude = np.mean(np.linalg.norm(accelerometer_data, axis=1))
        logger.debug(f"📏 Average Accel Magnitude: {avg_magnitude:.4f}")

        if avg_magnitude < ACCELERATION_THRESHOLD:
            predicted_class = "Stationary"
        else:
            predicted_class = load_model_and_predict(model, input_tensor)

        latest_prediction["predicted_activity"] = predicted_class
        logger.info(f"🎯 Predicted activity: {predicted_class}")
        return jsonify({"predicted_activity": predicted_class})

    except Exception as e:
        logger.error(f"❌ Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/latest", methods=["GET"])
def get_latest_prediction():
    return jsonify(latest_prediction)

def smooth_data(data, window_size=5):
    """
    Apply moving average filter
    """
    smoothed = np.zeros_like(data)
    for i in range(data.shape[1]):
        smoothed[:, i] = np.convolve(data[:, i], np.ones(window_size)/window_size, mode='same')
    return smoothed

def preprocess_input(data_series):
    """
    Convert sensor JSON to model input tensor
    """
    accelerometer_data = []

    for entry in data_series:
        if 'accelerometer' in entry['name']:
            values = entry['values']
            accelerometer_data.append([
                values.get('x', 0.0),
                values.get('y', 0.0),
                values.get('z', 0.0)
            ])

    # Fix sample length
    if len(accelerometer_data) < 128:
        accelerometer_data.extend([[0.0, 0.0, 0.0]] * (128 - len(accelerometer_data)))
    elif len(accelerometer_data) > 128:
        accelerometer_data = accelerometer_data[:128]

    accelerometer_data = np.array(accelerometer_data, dtype=np.float32)

    logger.debug(f"📉 Raw accel mean/std: {np.mean(accelerometer_data, axis=0)} / {np.std(accelerometer_data, axis=0)}")

    # Smooth
    accelerometer_data = smooth_data(accelerometer_data)
    logger.debug(f"📈 Smoothed accel mean/std: {np.mean(accelerometer_data, axis=0)} / {np.std(accelerometer_data, axis=0)}")

    # Normalize
    normalized = (accelerometer_data - accelerometer_data.mean(axis=0)) / (accelerometer_data.std(axis=0) + 1e-6)

    input_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 128, 3)
    return input_tensor, accelerometer_data

def load_model_and_predict(model, input_tensor):
    """
    Run model inference
    """
    with torch.no_grad():
        output = model(input_tensor)
        pred_class_idx = output.argmax(dim=1).item()

    activity_labels = ['Walking', 'Walking', 'Sitting', 'Standing', 'Upstairs', 'Downstairs']
    return activity_labels[pred_class_idx]

# This is only for local development
if __name__ == "__main__":
    app.run(debug=True)

# Render uses gunicorn to run this: gunicorn app:app

