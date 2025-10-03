import torch
import torch.nn as nn
import joblib
import json
import pandas as pd
import numpy as np
from collections import deque
import subprocess
import time
import re
import os
from modelnew import LSTMAutoencoder, LSTMClassifier

# --- Constants & File Paths ---
SEQUENCE_LENGTH = 5
AUTOENCODER_DISTANCES = {30, 35}
FEATURES = ["dl_cqi", "ul_rsrp", "cqi_diff", "cqi_rolling_std", "quality_divergence", "distance_ft"]
FEATURES_TO_SCALE = ["dl_cqi", "ul_rsrp", "quality_divergence", "cqi_diff", "cqi_rolling_std"]
AUTOENCODER_MODEL_FNAME = "autoencoder_model.pth"
CLASSIFIER_MODEL_FNAME = "classifier_model.pth"
AE_SCALER_FNAME = "ae_scaler.pth"
CLS_SCALER_FNAME = "cls_scaler.pth"
THRESHOLD_FNAME = "threshold_config.json"
SCALER_DIVERGENCE_FNAME = "scaler_divergence.pth"
LOG_FILE_PATH = "./gnb_session.log"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

SCRIPT_PATH_REL = "./restart_script.sh"
# Create a copy of the current environment
my_env = os.environ.copy()

# Set the DISPLAY variable for the new process
my_env["DISPLAY"] = ":0"

class FeatureProcessor:
    def __init__(self, divergence_scaler, buffer_size=20):
        self.distance_buffers = {}
        self.buffer_size = buffer_size
        self.divergence_scaler = divergence_scaler
        self.overload_value = 9.0

    def process(self, data_point):
        distance = int(data_point["distance_ft"])
        if distance not in self.distance_buffers:
            self.distance_buffers[distance] = deque(maxlen=self.buffer_size)

        self.distance_buffers[distance].append(data_point)
        buffer_df = pd.DataFrame(list(self.distance_buffers[distance]))
        
        buffer_df['ul_rsrp'] = pd.to_numeric(buffer_df['ul_rsrp'], errors="coerce")
        buffer_df["dl_cqi"] = pd.to_numeric(buffer_df["dl_cqi"], errors="coerce")

        buffer_df = buffer_df.copy()

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

def parse_numeric_value(value_str):
    s = str(value_str).lower().strip()
    multiplier = 1
    if s.endswith('k'):
        multiplier = 1000
        s = s[:-1]
    elif s.endswith('m'):
        multiplier = 1_000_000
        s = s[:-1]
    return int(float(s) * multiplier)

def parse_srsran_log_line(line: str):
    """Parse srsRAN gNB log line with optimized regex"""
    data_pattern = re.compile(r"""
        ^\s*(?P<pci>\d+)\s+
        (?P<rnti>\S+)\s*\|\s*
        (?P<dl_cqi>\S+)\s+
        (?P<dl_ri>\S+)\s+
        (?P<dl_mcs>\S+)\s+
        (?P<dl_brate>\S+)\s+
        (?P<dl_ok>\S+)\s+
        (?P<dl_nok>\S+)\s+
        (?P<dl_perc>\d+)%\s+
        (?P<dl_bs>\S+)\s*\|\s*
        (?P<pusch>\S+)\s+
        (?P<rsrp>\S+)\s+
        (?P<ul_ri>\S+)\s+
        (?P<ul_mcs>\S+)\s+
        (?P<ul_brate>\S+)\s+
        (?P<ul_ok>\S+)\s+
        (?P<ul_nok>\S+)\s+
        (?P<ul_perc>\d+)%\s+
        (?P<bsr>\S+)\s+
        (?P<ta>\S+)\s+
        (?P<phr>\S+)
    """, re.VERBOSE)
    match = data_pattern.match(line)
    if not match:
        return None

    data = match.groupdict()

    try:
        data['dl_cqi'] = 1 if str(data['dl_cqi']).lower() == 'n/a' else int(data['dl_cqi'])
        data['pci'] = 1 if str(data['pci']).lower() == 'n/a' else int(data['pci'])
        data['rnti'] = int(data['rnti'], 16)
        data['dl_ri'] = 1.0 if str(data['dl_ri']).lower() == 'n/a' else float(data['dl_ri'])
        data['ul_ri'] = int(data['ul_ri'])
        data['ul_mcs'] = int(data['ul_mcs'])
        data['bsr'] = parse_numeric_value(data['bsr'])
        
        # ### MODIFICATION 1: Parse RSRP to float or np.nan ###
        # Instead of converting 'n/a' to 9 here, we use np.nan to signify a missing value
        # which we will handle in the main loop logic.
        rsrp_val = str(data['rsrp']).lower()
        if rsrp_val in ('n/a', 'ovl'):
            data['ul_rsrp'] = np.nan
        else:
            data['ul_rsrp'] = float(rsrp_val)
        
        print(f"Data match found: {data}")
    except (ValueError, KeyError) as e:
        print(f"Value conversion or key error: {e}")
        return None

    return data

class FastLogReader:
    def __init__(self, log_file_path, buffer_size=8192):
        self.log_file_path = log_file_path
        self.buffer_size = buffer_size
        self.file_position = 0
        self.partial_line = ""
        self.ansi_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        
    def clean_line(self, line):
        clean = self.ansi_pattern.sub('', line)
        return ''.join(c for c in clean if ord(c) >= 32 or c == '\n').strip()

    def read_new_lines(self):
        while not os.path.exists(self.log_file_path):
            time.sleep(0.1)
            
        with open(self.log_file_path, 'r', encoding='utf-8', errors='ignore') as log_file:
            log_file.seek(0, 2)
            self.file_position = log_file.tell()
            
            while True:
                current_size = os.path.getsize(self.log_file_path)
                
                if current_size > self.file_position:
                    log_file.seek(self.file_position)
                    new_data = log_file.read(self.buffer_size)
                    
                    if new_data:
                        full_data = self.partial_line + new_data
                        lines = full_data.split('\n')
                        self.partial_line = lines[-1]
                        
                        for line in lines[:-1]:
                            cleaned_line = self.clean_line(line)
                            if cleaned_line and len(cleaned_line) > 20:
                                yield cleaned_line
                        
                        self.file_position = log_file.tell()
                
                time.sleep(0.01)

def create_model_input(parsed_gnb_data):
    return {
        "dl_cqi": parsed_gnb_data.get("dl_cqi"),
        "ul_rsrp": parsed_gnb_data.get("ul_rsrp"),
        "distance_ft": 10
    }

if __name__ == "__main__":
    jamming_mitigation_active = False
    last_restart_time = 0
    COOLDOWN_PERIOD_SECONDS = 300
 
    autoencoder_model, classifier_model, ae_scaler, cls_scaler, divergence_scaler, threshold_config = load_artifacts()
    
    anomaly_thresholds = threshold_config["thresholds"]
    global_threshold = threshold_config["global_threshold"]
    feature_processor = FeatureProcessor(divergence_scaler=divergence_scaler)
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    # ### MODIFICATION 2: Add state counter for initialization ###
    initial_rsrp_values_collected = 0
    
    log_reader = FastLogReader(LOG_FILE_PATH)
    print("Starting real-time log processing...")

    ANOMALY_DETECTED = False
    
    for log_line in log_reader.read_new_lines():
        try:
            gnb_data = parse_srsran_log_line(log_line)
            if not gnb_data:
                continue

            # ### MODIFICATION 3: Implement warm-up logic ###
            # Phase 1: Collect initial data points with valid RSRP
            if initial_rsrp_values_collected < SEQUENCE_LENGTH:
                # Check if ul_rsrp is a valid number (not NaN)
                if 'ul_rsrp' in gnb_data and not np.isnan(gnb_data['ul_rsrp']):
                    initial_rsrp_values_collected += 1
                    print(f"Collected initial valid RSRP point ({initial_rsrp_values_collected}/{SEQUENCE_LENGTH})...")
                else:
                    # If RSRP is missing during initialization, skip this data point
                    print("Waiting for a valid RSRP value to initialize the system...")
                    continue
            
            # Phase 2: Operational mode. If RSRP is missing now, it's an anomaly.
            # Convert NaN to the overload value (9.0) for the model.
            if np.isnan(gnb_data['ul_rsrp']):
                gnb_data['ul_rsrp'] = 9.0

            # Convert to model input format
            model_input = create_model_input(gnb_data)
            
            print("Processing data point:", model_input)
            processed_features = feature_processor.process(model_input)
            if processed_features is None:
                continue

            sequence_buffer.append(processed_features)
            
            # Ensure we don't start predicting until the buffer is full
            if len(sequence_buffer) < SEQUENCE_LENGTH:
                # This check also implicitly covers the initialization phase
                print(f"Filling buffer... {len(sequence_buffer)}/{SEQUENCE_LENGTH}")
                continue

            # --- The rest of the prediction logic remains the same ---
            sequence_list = list(sequence_buffer)
            current_distance = int(sequence_list[-1]["distance_ft"])
            features_df = pd.DataFrame(sequence_list)[FEATURES]

            with torch.no_grad():
                if current_distance in AUTOENCODER_DISTANCES:
                    scaled_df = features_df.copy()
                    scaled_df[FEATURES_TO_SCALE] = ae_scaler.transform(scaled_df[FEATURES_TO_SCALE])
                    
                    input_tensor = torch.tensor(scaled_df.values, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    reconstructed_output = autoencoder_model(input_tensor)
                    reconstruction_error = torch.mean((reconstructed_output - input_tensor) ** 2).item()
                    
                    anomaly_threshold = anomaly_thresholds.get(str(current_distance), global_threshold)
                    if reconstruction_error > anomaly_threshold:
                        print(f"ANOMALY DETECTED at {current_distance}ft! Error: {reconstruction_error:.4f}")
                        if not jamming_mitigation_active:
                            print("==================Jamming detected! Triggering restart to anti-jamming config...===============")
                            subprocess.run([SCRIPT_PATH_REL, "restart_jamming"])
                            jamming_mitigation_active = True
                            last_restart_time = time.time()
                    else:
                        print(f"Normal at {current_distance}ft. Error: {reconstruction_error:.4f}")
                else:
                    scaled_df = features_df.copy()
                    scaled_df[FEATURES_TO_SCALE] = cls_scaler.transform(scaled_df[FEATURES_TO_SCALE])
                    
                    input_tensor = torch.tensor(scaled_df.values, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    classifier_output = classifier_model(input_tensor)
                    anomaly_probability = torch.sigmoid(classifier_output).item()
                    
                    if anomaly_probability > 0.8:
                        print(f"===========ANOMALY DETECTED at {current_distance}ft! Confidence: {anomaly_probability:.2f}==============")
                        if not jamming_mitigation_active:
                            print("Jamming detected! Triggering restart to anti-jamming config...")
                            subprocess.run([SCRIPT_PATH_REL, "restart_jamming"])
                            jamming_mitigation_active = True
                            last_restart_time = time.time()
                    else:
                        print(f"Normal at {current_distance}ft. Confidence: {anomaly_probability:.2f}")
        except Exception as e:
            print(f"Error processing log line: {e}")
            continue
