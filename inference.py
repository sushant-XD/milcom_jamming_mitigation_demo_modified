import torch
import torch.nn as nn
import joblib
import json
import pandas as pd
import numpy as np
from collections import deque
import zmq
from modelnew import LSTMAutoencoder, LSTMClassifier

SEQUENCE_LENGTH = 10
AUTOENCODER_DISTANCES = {30, 35}

FEATURES = [
    "dl_cqi",
    "ul_rsrp",
    "cqi_diff",
    "cqi_rolling_std",
    "quality_divergence",
    "distance_ft",
]
FEATURES_TO_SCALE = [
    "dl_cqi",
    "ul_rsrp",
    "quality_divergence",
    "cqi_diff",
    "cqi_rolling_std",
]
AUTOENCODER_MODEL_FNAME = "autoencoder_model.pth"
CLASSIFIER_MODEL_FNAME = "classifier_model.pth"
AE_SCALER_FNAME = "ae_scaler.pth"
CLS_SCALER_FNAME = "cls_scaler.pth"
THRESHOLD_FNAME = "threshold_config.json"
SCALER_DIVERGENCE_FNAME = "scaler_divergence.pth"

ZMQ_SOCKET_URL = "tcp://10.28.28.12:5555"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

test_kpm_data = json.load(open("test_data.json"))


class FeatureProcessor:
    def __init__(self, divergence_scaler, buffer_size=20):
        self.buffers = {}
        self.buffer_size = buffer_size
        self.divergence_scaler = divergence_scaler
        self.overload_value = -40.0

    def process(self, data_point):
        dist = int(data_point["distance_ft"])
        if dist not in self.buffers:
            self.buffers[dist] = deque(maxlen=self.buffer_size)

        self.buffers[dist].append(data_point)
        temp_df = pd.DataFrame(list(self.buffers[dist]))

        
        temp_df["ul_rsrp"] = np.where(
            temp_df["ul_rsrp"] == "ovl", self.overload_value, temp_df["ul_rsrp"]
        )
        temp_df["ul_rsrp"] = np.where(
            temp_df["ul_rsrp"] == "n/a", self.overload_value, temp_df["ul_rsrp"]
        )
        temp_df['ul_rsrp'] = pd.to_numeric(temp_df['ul_rsrp'], errors="coerce")
        temp_df["dl_cqi"] = pd.to_numeric(temp_df["dl_cqi"], errors="coerce")

        # Fill any remaining NaN values with default values
        temp_df = temp_df.copy()  # Ensure we have a proper DataFrame, not a view
        temp_df['ul_rsrp'] = temp_df['ul_rsrp'].fillna(self.overload_value)
        temp_df['dl_cqi'] = temp_df['dl_cqi'].fillna(0)

        # Ensure we have valid data for scaling
        if temp_df[["dl_cqi", "ul_rsrp"]].isna().any().any():
            print("Warning: NaN values still present after preprocessing")
            return None

        try:
            scaled_base = self.divergence_scaler.transform(temp_df[["dl_cqi", "ul_rsrp"]])
            temp_df["quality_divergence"] = scaled_base[:, 1] - scaled_base[:, 0]
            temp_df["cqi_diff"] = temp_df["dl_cqi"].diff().fillna(0)
            temp_df["cqi_rolling_std"] = (
                temp_df["dl_cqi"].rolling(window=5, min_periods=1).std().fillna(0)
            )
        except Exception as e:
            print(f"Error in feature processing: {e}")
            return None

        return temp_df.iloc[-1].to_dict()
def load_artifacts():
    print("loading model artifacts")

    model_ae = LSTMAutoencoder(
        input_size=len(FEATURES), sequence_len=SEQUENCE_LENGTH
    ).to(DEVICE)
    model_ae.load_state_dict(torch.load(AUTOENCODER_MODEL_FNAME))
    model_ae.eval()

    model_cls = LSTMClassifier(input_size=len(FEATURES)).to(DEVICE)
    model_cls.load_state_dict(torch.load(CLASSIFIER_MODEL_FNAME))
    model_cls.eval()

    ae_scaler = joblib.load(AE_SCALER_FNAME)
    cls_scaler = joblib.load(CLS_SCALER_FNAME)
    scaler_divergence = joblib.load(SCALER_DIVERGENCE_FNAME)

    with open(THRESHOLD_FNAME, "r") as f:
        threshold = json.load(f)
    print("Artifacts loaded")
    return model_ae, model_cls, ae_scaler, cls_scaler, scaler_divergence, threshold


def subscribe_to_kpm_data():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(ZMQ_SOCKET_URL)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    print(f"ZMQ Setup complete on {ZMQ_SOCKET_URL}")
    while True:
        message = socket.recv_string()
        try:
            _topic, payload_str = message.split(" ", 1)
            data_point = json.loads(payload_str)
            print("KPM data received:")
            print(data_point)
            yield data_point
        except ValueError as e:
            print(f"Value Error while decoding. Test message received")
            print(payload_str)


if __name__ == "__main__":
    model_ae, model_cls, ae_scaler, cls_scaler, divergence_scaler, threshold_config = (
        load_artifacts()
    )
    thresholds = threshold_config["thresholds"]
    global_threshold = threshold_config["global_threshold"]
    feature_processor = FeatureProcessor(divergence_scaler=divergence_scaler)
    data_buffer = deque(maxlen=SEQUENCE_LENGTH)
    print("Reading kpm data")
    for raw_data_point in test_kpm_data:
        try:
            print(raw_data_point)
            point = feature_processor.process(raw_data_point)
            if point is None:
                continue

            data_buffer.append(point)
            if len(data_buffer) < SEQUENCE_LENGTH:
                continue

            sequence_list = list(data_buffer)
            current_distance = int(sequence_list[-1]["distance_ft"])
            sequence_df = pd.DataFrame(sequence_list)[FEATURES]

            # todo: need to get and parse the KPM data
            with torch.no_grad():
                if current_distance in AUTOENCODER_DISTANCES:
                    temp_df = sequence_df.copy()
                    temp_df[FEATURES_TO_SCALE] = ae_scaler.transform(
                        temp_df[FEATURES_TO_SCALE]
                    )
                    seq_tensor = (
                        torch.tensor(temp_df.values, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(DEVICE)
                    )
                    reconstructed = model_ae(seq_tensor)
                    error = torch.mean((reconstructed - seq_tensor) ** 2).item()
                    threshold = thresholds.get(str(current_distance), global_threshold)
                    if error > threshold:
                        print(f"Anomaly Detected!!!!! Confidence: {error}")
                    else:
                        print(f"No Anomaly Detected. Confidence: {error}")

                else:
                    temp_df = sequence_df.copy()
                    temp_df[FEATURES_TO_SCALE] = cls_scaler.transform(
                        temp_df[FEATURES_TO_SCALE]
                    )
                    seq_tensor = (
                        torch.tensor(temp_df.values, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(DEVICE)
                    )
                    output = model_cls(seq_tensor)
                    probability = torch.sigmoid(output).item()
                    if probability > 0.5:
                        print(f"Anomaly detected. Confidence: {probability:.2f}")
                    else:
                        print(f"probability: {probability}")

        except Exception as e:
            print(f"Exception: {e}")
