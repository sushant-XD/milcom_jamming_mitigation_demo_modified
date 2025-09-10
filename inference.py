import torch
import torch.nn as nn
import joblib
import json
import pandas as pd
import numpy as np
from collections import deque
import time
import re
import os
from modelnew import LSTMAutoencoder, LSTMClassifier

# --- Constants & File Paths ---
SEQUENCE_LENGTH = 10
AUTOENCODER_DISTANCES = {30, 35}
FEATURES = ["dl_cqi", "ul_rsrp", "cqi_diff", "cqi_rolling_std", "quality_divergence", "distance_ft"]
FEATURES_TO_SCALE = ["dl_cqi", "ul_rsrp", "quality_divergence", "cqi_diff", "cqi_rolling_std"]
AUTOENCODER_MODEL_FNAME = "autoencoder_model.pth"
CLASSIFIER_MODEL_FNAME = "classifier_model.pth"
AE_SCALER_FNAME = "ae_scaler.pth"
CLS_SCALER_FNAME = "cls_scaler.pth"
THRESHOLD_FNAME = "threshold_config.json"
SCALER_DIVERGENCE_FNAME = "scaler_divergence.pth"
LOG_FILE_PATH = "../ran-tester-ue/gnb_session.log"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class FeatureProcessor:
    def __init__(self, divergence_scaler, buffer_size=20):
        self.distance_buffers = {}
        self.buffer_size = buffer_size
        self.divergence_scaler = divergence_scaler
        self.overload_value = -40.0

    def process(self, data_point):
        distance = int(data_point["distance_ft"])
        if distance not in self.distance_buffers:
            self.distance_buffers[distance] = deque(maxlen=self.buffer_size)

        self.distance_buffers[distance].append(data_point)
        buffer_df = pd.DataFrame(list(self.distance_buffers[distance]))

        buffer_df["ul_rsrp"] = np.where(
            buffer_df["ul_rsrp"] == "ovl", self.overload_value, buffer_df["ul_rsrp"]
        )
        buffer_df["ul_rsrp"] = np.where(
            buffer_df["ul_rsrp"] == "n/a", self.overload_value, buffer_df["ul_rsrp"]
        )
        buffer_df['ul_rsrp'] = pd.to_numeric(buffer_df['ul_rsrp'], errors="coerce")
        buffer_df["dl_cqi"] = pd.to_numeric(buffer_df["dl_cqi"], errors="coerce")

        buffer_df = buffer_df.copy()
        buffer_df['ul_rsrp'] = buffer_df['ul_rsrp'].fillna(self.overload_value)
        buffer_df['dl_cqi'] = buffer_df['dl_cqi'].fillna(0)

        if buffer_df[["dl_cqi", "ul_rsrp"]].isna().any().any():
            print("Warning: NaN values still present after preprocessing")
            return None

        try:
            scaled_features = self.divergence_scaler.transform(buffer_df[["dl_cqi", "ul_rsrp"]])
            buffer_df["quality_divergence"] = scaled_features[:, 1] - scaled_features[:, 0]
            buffer_df["cqi_diff"] = buffer_df["dl_cqi"].diff().fillna(0)
            buffer_df["cqi_rolling_std"] = (
                buffer_df["dl_cqi"].rolling(window=5, min_periods=1).std().fillna(0)
            )
        except Exception as e:
            print(f"Error in feature processing: {e}")
            return None

        return buffer_df.iloc[-1].to_dict()

def load_artifacts():
    print("Loading model artifacts...")
    autoencoder_model = LSTMAutoencoder(
        input_size=len(FEATURES), sequence_len=SEQUENCE_LENGTH
    ).to(DEVICE)
    autoencoder_model.load_state_dict(torch.load(AUTOENCODER_MODEL_FNAME))
    autoencoder_model.eval()

    classifier_model = LSTMClassifier(input_size=len(FEATURES)).to(DEVICE)
    classifier_model.load_state_dict(torch.load(CLASSIFIER_MODEL_FNAME))
    classifier_model.eval()

    ae_scaler = joblib.load(AE_SCALER_FNAME)
    cls_scaler = joblib.load(CLS_SCALER_FNAME)
    divergence_scaler = joblib.load(SCALER_DIVERGENCE_FNAME)

    with open(THRESHOLD_FNAME, "r") as f:
        threshold_config = json.load(f)
    print("Artifacts loaded.")
    return autoencoder_model, classifier_model, ae_scaler, cls_scaler, divergence_scaler, threshold_config

def parse_srsran_log_line(line: str):
    """Parse srsRAN gNB log line with optimized regex"""
    # Debug: print the line being parsed
    print(f"Attempting to parse: {repr(line[:100])}")
   
    data_pattern = re.compile(r"""
        ^\s*(?P<pci>\d+)\s+
        (?P<rnti>\d+)\s*\|\s*
        (?P<dl_cqi>\d+)\s+
        (?P<dl_ri>[\d.]+)\s+
        (?P<dl_mcs>\d+)\s+
        (?P<dl_brate>\d+)\s+
        (?P<dl_ok>\d+)\s+
        (?P<dl_nok>\d+)\s+
        (?P<dl_perc>\d+)%\s+
        (?P<dl_bs>\d+)\s*\|\s*
        (?P<pusch>\S+)\s+
        (?P<rsrp>\S+)\s+
        (?P<ul_ri>\d+)\s+
        (?P<ul_mcs>\d+)\s+
        (?P<ul_brate>\d+)\s+
        (?P<ul_ok>\d+)\s+
        (?P<ul_nok>\d+)\s+
        (?P<ul_perc>\d+)%\s+
        (?P<bsr>\d+)\s+
        (?P<ta>\S+)\s+
        (?P<phr>\S+)
    """, re.VERBOSE)
    match = data_pattern.match(line)
    if not match:
        print(f"No regex match for line: {line[:100]}...")
        return None

    # Extract all named groups as dictionary
    data = match.groupdict()

    # Optional: parse/convert some values
    try:
        data['pci'] = int(data['pci'])
        data['rnti'] = int(data['rnti'])
        data['dl_cqi'] = int(data['dl_cqi'])
        data['dl_ri'] = float(data['dl_ri'])
        data['ul_ri'] = int(data['ul_ri'])
        data['ul_mcs'] = int(data['ul_mcs'])
        data['bsr'] = int(data['bsr'])
        print(f"Data match found: {data}")
    except ValueError as e:
        print(f"Value conversion error: {e}")
        return None

    return data

class FastLogReader:
    def __init__(self, log_file_path, buffer_size=8192):
        self.log_file_path = log_file_path
        self.buffer_size = buffer_size
        self.file_position = 0
        self.partial_line = ""
        # Compiled ANSI escape sequence pattern
        self.ansi_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        
    def clean_line(self, line):
        """Fast line cleaning"""
        clean = self.ansi_pattern.sub('', line)
        return ''.join(c for c in clean if ord(c) >= 32 or c == '\n').strip()

    def read_new_lines(self):
        """Generator that yields new lines as they're written to the file"""
        # Wait for file to exist
        while not os.path.exists(self.log_file_path):
            time.sleep(0.1)
            
        with open(self.log_file_path, 'r', encoding='utf-8', errors='ignore') as log_file:
            # Start from end if file already exists
            log_file.seek(0, 2)
            self.file_position = log_file.tell()
            
            while True:
                current_size = os.path.getsize(self.log_file_path)
                
                if current_size > self.file_position:
                    log_file.seek(self.file_position)
                    new_data = log_file.read(self.buffer_size)
                    
                    if new_data:
                        # Handle partial lines
                        full_data = self.partial_line + new_data
                        lines = full_data.split('\n')
                        self.partial_line = lines[-1]
                        
                        # Process complete lines
                        for line in lines[:-1]:
                            cleaned_line = self.clean_line(line)
                            if cleaned_line and len(cleaned_line) > 20:  # Skip short lines
                                yield cleaned_line
                        
                        self.file_position = log_file.tell()
                
                time.sleep(0.01)  # 10ms polling

def create_model_input(parsed_gnb_data):
    """Convert parsed gNB data to model input format"""
    return {
        "dl_cqi": parsed_gnb_data.get("dl_cqi"),
        "ul_rsrp": parsed_gnb_data.get("ul_rsrp"),
        "distance_ft": 30  # TODO: Make this configurable
    }

if __name__ == "__main__":
    autoencoder_model, classifier_model, ae_scaler, cls_scaler, divergence_scaler, threshold_config = load_artifacts()
    
    anomaly_thresholds = threshold_config["thresholds"]
    global_threshold = threshold_config["global_threshold"]
    feature_processor = FeatureProcessor(divergence_scaler=divergence_scaler)
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    # Initialize fast log reader (set start_from_beginning=True for testing)
    log_reader = FastLogReader(LOG_FILE_PATH)
    
    print("Starting real-time log processing...")
    for log_line in log_reader.read_new_lines():
        try:
            # Parse the log line
            gnb_data = parse_srsran_log_line(log_line)
            if not gnb_data:
                continue
                
            # Convert to model input format
            model_input = create_model_input(gnb_data)
            
            print("Processing data point:", model_input)
            processed_features = feature_processor.process(model_input)
            if processed_features is None:
                continue

            sequence_buffer.append(processed_features)
            if len(sequence_buffer) < SEQUENCE_LENGTH:
                continue

            # Prepare data for model inference
            sequence_list = list(sequence_buffer)
            current_distance = int(sequence_list[-1]["distance_ft"])
            features_df = pd.DataFrame(sequence_list)[FEATURES]

            with torch.no_grad():
                if current_distance in AUTOENCODER_DISTANCES:
                    # Autoencoder anomaly detection
                    scaled_df = features_df.copy()
                    scaled_df[FEATURES_TO_SCALE] = ae_scaler.transform(scaled_df[FEATURES_TO_SCALE])
                    
                    input_tensor = torch.tensor(scaled_df.values, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    reconstructed_output = autoencoder_model(input_tensor)
                    reconstruction_error = torch.mean((reconstructed_output - input_tensor) ** 2).item()
                    
                    anomaly_threshold = anomaly_thresholds.get(str(current_distance), global_threshold)
                    if reconstruction_error > anomaly_threshold:
                        print(f"ANOMALY DETECTED at {current_distance}ft! Error: {reconstruction_error:.4f}")
                    else:
                        print(f"Normal at {current_distance}ft. Error: {reconstruction_error:.4f}")
                else:
                    # Classifier anomaly detection
                    scaled_df = features_df.copy()
                    scaled_df[FEATURES_TO_SCALE] = cls_scaler.transform(scaled_df[FEATURES_TO_SCALE])
                    
                    input_tensor = torch.tensor(scaled_df.values, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    classifier_output = classifier_model(input_tensor)
                    anomaly_probability = torch.sigmoid(classifier_output).item()
                    
                    if anomaly_probability > 0.5:
                        print(f"ANOMALY DETECTED at {current_distance}ft! Confidence: {anomaly_probability:.2f}")
                    else:
                        print(f"Normal at {current_distance}ft. Confidence: {anomaly_probability:.2f}")

        except Exception as e:
            print(f"Error processing log line: {e}")
            continue
