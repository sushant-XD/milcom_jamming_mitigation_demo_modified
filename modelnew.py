import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

# --- Configuration ---
JAMMING_DATA_DIR = "jamming_processed_data"
NO_JAMMING_DATA_DIR = "no_jamming_processed_data"
SEQUENCE_LENGTH = 10
BATCH_SIZE = 64
EPOCHS = 30000
LEARNING_RATE_AE = 0.0005
LEARNING_RATE_CLS = 0.001
DROPOUT_RATE = 0.3
EARLY_STOPPING_PATIENCE = 15
THRESHOLD_PERCENTILE = 92
AUTOENCODER_DISTANCES = {30, 35}

FEATURES = ['dl_cqi', 'ul_rsrp', 'cqi_diff', 'cqi_rolling_std', 'quality_divergence', 'distance_ft']
FEATURES_TO_SCALE = ['dl_cqi', 'ul_rsrp', 'quality_divergence', 'cqi_diff', 'cqi_rolling_std']

# --- Helper Functions and Model Definitions ---
def load_and_process_csvs_from_folder(folder_path):
    if not os.path.isdir(folder_path): print(f"Error: Directory not found at '{folder_path}'"); return None
    all_dfs = []; print(f"Searching for all '.csv' files in '{folder_path}'...")
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.csv'):
                file_path = os.path.join(root, filename)
                try: df = pd.read_csv(file_path, low_memory=False); all_dfs.append(df)
                except Exception as e: print(f"  -> Could not read or process {filename}. Error: {e}")
    if not all_dfs: print(f"  -> No non-empty CSV files found in '{folder_path}'."); return None
    return pd.concat(all_dfs, ignore_index=True)

def create_sequences_with_distance(df, features, sequence_length, distance_col='distance_ft'):
    data_values = df[features].values; sequences = [];
    if len(data_values) <= sequence_length: return np.array(sequences), np.array([])
    for i in range(len(data_values) - sequence_length): sequences.append(data_values[i:(i + sequence_length)])
    sequences = np.array(sequences);
    if len(sequences) == 0: return sequences, np.array([])
    distances = [df[distance_col].iloc[i + sequence_length - 1] for i in range(len(data_values) - sequence_length)]
    return sequences, np.array(distances)

class AutoencoderDataset(Dataset):
    def __init__(self, sequences): self.sequences = torch.tensor(sequences, dtype=torch.float32)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.sequences[idx]

class ClassifierDataset(Dataset):
    def __init__(self, sequences, labels): self.sequences = torch.tensor(sequences, dtype=torch.float32); self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.labels[idx]

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, sequence_len, hidden_size=128, latent_size=32, dropout_rate=0.2):
        super(LSTMAutoencoder, self).__init__(); self.encoder_lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True); self.encoder_dropout1 = nn.Dropout(dropout_rate); self.encoder_relu1 = nn.ReLU(); self.encoder_lstm2 = nn.LSTM(hidden_size, latent_size, batch_first=True); self.encoder_dropout2 = nn.Dropout(dropout_rate); self.encoder_relu2 = nn.ReLU(); self.decoder_lstm1 = nn.LSTM(latent_size, hidden_size, batch_first=True); self.decoder_dropout3 = nn.Dropout(dropout_rate); self.decoder_relu1 = nn.ReLU(); self.decoder_lstm2 = nn.LSTM(hidden_size, input_size, batch_first=True)
    
    # ### CORRECTED AND READABLE FORWARD METHOD ###
    def forward(self, x):
        # Encoder
        out_lstm1, _ = self.encoder_lstm1(x)
        out_dropout1 = self.encoder_dropout1(out_lstm1)
        out_relu1 = self.encoder_relu1(out_dropout1)
        
        out_lstm2, (h_n_encoder, c_n_encoder) = self.encoder_lstm2(out_relu1)
        # Corrected line: input is out_lstm2
        out_dropout2 = self.encoder_dropout2(out_lstm2)
        encoded_sequence_activated = self.encoder_relu2(out_dropout2)
        
        # Latent Space
        latent_vector = encoded_sequence_activated[:, -1, :]
        decoder_input_sequence = latent_vector.unsqueeze(1).repeat(1, x.size(1), 1)
        
        # Decoder
        out_lstm3, _ = self.decoder_lstm1(decoder_input_sequence)
        out_dropout3 = self.decoder_dropout3(out_lstm3)
        out_relu3 = self.decoder_relu1(out_dropout3)
        
        reconstructed_seq, _ = self.decoder_lstm2(out_relu3)
        return reconstructed_seq

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout_rate=0.3):
        super(LSTMClassifier, self).__init__(); self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate); self.fc1 = nn.Linear(hidden_size, hidden_size // 2); self.relu = nn.ReLU(); self.dropout = nn.Dropout(dropout_rate); self.fc2 = nn.Linear(hidden_size // 2, 1)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x); out = h_n[-1, :, :]; out = self.fc1(out); out = self.relu(out); out = self.dropout(out); out = self.fc2(out); return out.squeeze(-1)

# --- Main Execution Block ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normal_df_orig = load_and_process_csvs_from_folder(NO_JAMMING_DATA_DIR)
    jammed_df_orig = load_and_process_csvs_from_folder(JAMMING_DATA_DIR)
    if normal_df_orig is None or jammed_df_orig is None: exit()
    
    normal_df = normal_df_orig.copy()
    jammed_df = jammed_df_orig.copy()
    
    # --- 1. Data Preparation ---
    print("\n" + "="*80 + "\n=== PREPARING DATA FOR BOTH MODELS ===" + "\n" + "="*80 + "\n")
    temp_normal_rsrp = pd.to_numeric(normal_df['ul_rsrp'], errors='coerce'); temp_jammed_rsrp = pd.to_numeric(jammed_df['ul_rsrp'], errors='coerce'); max_rsrp = pd.concat([temp_normal_rsrp, temp_jammed_rsrp]).max(); overload_value = max_rsrp + 10
    print(f"Max measured RSRP: {max_rsrp:.2f}. Using {overload_value:.2f} for 'ovl'.")
    temp_normal_clean = normal_df.copy(); temp_normal_clean['ul_rsrp'].replace('ovl', overload_value, inplace=True); temp_normal_clean['ul_rsrp'] = pd.to_numeric(temp_normal_clean['ul_rsrp'], errors='coerce'); temp_normal_clean['dl_cqi'] = pd.to_numeric(temp_normal_clean['dl_cqi'], errors='coerce'); temp_normal_clean.dropna(subset=['ul_rsrp', 'dl_cqi'], inplace=True)
    scaler_divergence = RobustScaler().fit(temp_normal_clean[['dl_cqi', 'ul_rsrp']])
    joblib.dump(scaler_divergence, "scaler_divergence.pth")
    for df in [normal_df, jammed_df]:
        df['ul_rsrp'].replace('ovl', overload_value, inplace=True); df['ul_rsrp'] = pd.to_numeric(df['ul_rsrp'], errors='coerce'); df['ul_rsrp'].ffill(inplace=True); df['ul_rsrp'].bfill(inplace=True); df['distance_ft'] = pd.to_numeric(df['distance_ft'], errors='coerce'); df['dl_cqi'] = pd.to_numeric(df['dl_cqi'], errors='coerce');
        df.dropna(subset=['distance_ft', 'dl_cqi', 'ul_rsrp'], inplace=True); df['distance_ft'] = df['distance_ft'].astype(int);
        scaled_base = scaler_divergence.transform(df[['dl_cqi', 'ul_rsrp']]); df['quality_divergence'] = scaled_base[:, 1] - scaled_base[:, 0];
        df.sort_values(by='distance_ft', kind='mergesort', inplace=True); df['cqi_diff'] = df.groupby('distance_ft')['dl_cqi'].diff().fillna(0); df['cqi_rolling_std'] = df.groupby('distance_ft')['dl_cqi'].rolling(window=5).std().fillna(0).reset_index(level=0, drop=True); df.dropna(subset=FEATURES, inplace=True)
    
    # --- 2. Train the LSTM Autoencoder ---
    print("\n" + "="*80 + "\n=== TRAINING AUTOENCODER MODEL ===" + "\n" + "="*80 + "\n")
    ae_scaler = RobustScaler().fit(normal_df[FEATURES_TO_SCALE])
    normal_df_ae = normal_df.copy(); normal_df_ae[FEATURES_TO_SCALE] = ae_scaler.transform(normal_df_ae[FEATURES_TO_SCALE])
    normal_sequences_ae, normal_distances_ae = create_sequences_with_distance(normal_df_ae, FEATURES, SEQUENCE_LENGTH)
    train_seq_ae, val_seq_ae, _, val_dist_ae = train_test_split(normal_sequences_ae, normal_distances_ae, test_size=0.2, random_state=42)
    train_loader_ae = DataLoader(AutoencoderDataset(train_seq_ae), batch_size=BATCH_SIZE, shuffle=True)
    val_loader_ae = DataLoader(AutoencoderDataset(val_seq_ae), batch_size=BATCH_SIZE, shuffle=False)
    model_ae = LSTMAutoencoder(input_size=len(FEATURES), sequence_len=SEQUENCE_LENGTH).to(device)
    criterion_ae = nn.MSELoss(); optimizer_ae = torch.optim.Adam(model_ae.parameters(), lr=LEARNING_RATE_AE)
    best_val_loss, epochs_no_improve = float('inf'), 0
    joblib.dump(ae_scaler,"ae_scaler.pth")
    for epoch in range(EPOCHS):
        model_ae.train();
        for seq, _ in train_loader_ae:
            seq_on_device = seq.to(device); reconstructed = model_ae(seq_on_device); loss = criterion_ae(reconstructed, seq_on_device); optimizer_ae.zero_grad(); loss.backward(); optimizer_ae.step()
        model_ae.eval()
        with torch.no_grad(): val_loss = np.mean([criterion_ae(model_ae(s.to(device)), s.to(device)).item() for s,_ in val_loader_ae])
        if val_loss < best_val_loss: best_val_loss, epochs_no_improve = val_loss, 0; torch.save(model_ae.state_dict(), "autoencoder_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE: print(f"AE training stopped early at epoch {epoch+1}"); break
    model_ae.load_state_dict(torch.load("autoencoder_model.pth")); model_ae.eval()
    val_errors_ae = [];
    with torch.no_grad():
        for seq, _ in val_loader_ae: reconstructed = model_ae(seq.to(device)); loss_per_item = torch.mean((reconstructed - seq.to(device)) ** 2, dim=[1, 2]); val_errors_ae.extend(loss_per_item.cpu().numpy())
    val_errors_ae = np.array(val_errors_ae); global_threshold = np.percentile(val_errors_ae, THRESHOLD_PERCENTILE); thresholds = {}
    for dist in set(normal_distances_ae):
        dist_errors = val_errors_ae[val_dist_ae == dist]
        if len(dist_errors) > 0: thresholds[dist] = np.percentile(dist_errors, THRESHOLD_PERCENTILE)

    global_thres = str(global_threshold)
    threshold_config = {
        "global_threshold": global_thres,
        #"per_distance": per_distance,
    }
    with open("threshold_config.json","w") as file:
        json.dump(threshold_config,file,indent=4)

    # --- 3. Train the LSTM Classifier ---
    print("\n" + "="*80 + "\n=== TRAINING CLASSIFIER MODEL ===" + "\n" + "="*80 + "\n")
    cls_scaler = RobustScaler().fit(pd.concat([normal_df, jammed_df])[FEATURES_TO_SCALE])
    normal_df_cls = normal_df.copy(); normal_df_cls[FEATURES_TO_SCALE] = cls_scaler.transform(normal_df_cls[FEATURES_TO_SCALE])
    jammed_df_cls = jammed_df.copy(); jammed_df_cls[FEATURES_TO_SCALE] = cls_scaler.transform(jammed_df_cls[FEATURES_TO_SCALE])
    norm_seq_cls, _ = create_sequences_with_distance(normal_df_cls, FEATURES, SEQUENCE_LENGTH)
    jamm_seq_cls, _ = create_sequences_with_distance(jammed_df_cls, FEATURES, SEQUENCE_LENGTH)
    all_seq_cls = np.concatenate([norm_seq_cls, jamm_seq_cls]); all_lbl_cls = np.concatenate([np.zeros(len(norm_seq_cls)), np.ones(len(jamm_seq_cls))])
    X_train, X_val, y_train, y_val = train_test_split(all_seq_cls, all_lbl_cls, test_size=0.2, random_state=42, stratify=all_lbl_cls)
    train_loader_cls = DataLoader(ClassifierDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader_cls = DataLoader(ClassifierDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    model_cls = LSTMClassifier(input_size=len(FEATURES)).to(device)
    criterion_cls = nn.BCEWithLogitsLoss(); optimizer_cls = torch.optim.Adam(model_cls.parameters(), lr=LEARNING_RATE_CLS)
    best_val_loss, epochs_no_improve = float('inf'), 0
    joblib.dump(cls_scaler,"cls_scaler.pth")
    for epoch in range(EPOCHS):
        model_cls.train(); batch_losses = []
        for seq, labels in train_loader_cls: seq, labels = seq.to(device), labels.to(device); optimizer_cls.zero_grad(); outputs = model_cls(seq); loss = criterion_cls(outputs, labels); loss.backward(); optimizer_cls.step(); batch_losses.append(loss.item())
        model_cls.eval(); val_losses = []
        with torch.no_grad():
            for seq, labels in val_loader_cls: seq, labels = seq.to(device), labels.to(device); outputs = model_cls(seq); loss = criterion_cls(outputs, labels); val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss: best_val_loss, epochs_no_improve = val_loss, 0; torch.save(model_cls.state_dict(), "classifier_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE: print(f"Classifier training stopped early at epoch {epoch+1}"); break

    # --- 4. Final Evaluation using the Meta-Model ---
    print("\n" + "="*80 + "\n=== EVALUATING META-MODEL SYSTEM ===" + "\n" + "="*80 + "\n")
    model_ae.load_state_dict(torch.load("autoencoder_model.pth")); model_ae.eval()
    model_cls.load_state_dict(torch.load("classifier_model.pth")); model_cls.eval()
    
    all_norm_seq_final, all_norm_dist_final = create_sequences_with_distance(normal_df, FEATURES, SEQUENCE_LENGTH)
    all_jamm_seq_final, all_jamm_dist_final = create_sequences_with_distance(jammed_df, FEATURES, SEQUENCE_LENGTH)
    all_sequences = np.concatenate([all_norm_seq_final, all_jamm_seq_final])
    true_labels = np.concatenate([np.zeros(len(all_norm_seq_final)), np.ones(len(all_jamm_seq_final))])
    all_distances = np.concatenate([all_norm_dist_final, all_jamm_dist_final])
    
    predictions = []
    with torch.no_grad():
        for i in range(len(all_sequences)):
            distance = all_distances[i]
            sequence = all_sequences[i] # This is a (10, 6) numpy array
            
            if distance in AUTOENCODER_DISTANCES:
                temp_df = pd.DataFrame(sequence, columns=FEATURES); temp_df[FEATURES_TO_SCALE] = ae_scaler.transform(temp_df[FEATURES_TO_SCALE]); seq_tensor = torch.tensor(temp_df.values, dtype=torch.float32).unsqueeze(0).to(device)
                reconstructed = model_ae(seq_tensor); error = torch.mean((reconstructed - seq_tensor) ** 2).item(); threshold = thresholds.get(distance, global_threshold); predictions.append(1 if error > threshold else 0)
            else:
                temp_df = pd.DataFrame(sequence, columns=FEATURES); temp_df[FEATURES_TO_SCALE] = cls_scaler.transform(temp_df[FEATURES_TO_SCALE]); seq_tensor = torch.tensor(temp_df.values, dtype=torch.float32).unsqueeze(0).to(device)
                output = model_cls(seq_tensor); prediction = 1 if torch.sigmoid(output).item() > 0.5 else 0; predictions.append(prediction)

    predictions = np.array(predictions)
    print("--- Overall Meta-Model Performance ---")
    print(classification_report(true_labels, predictions, target_names=['Normal', 'Anomaly'], labels=[0, 1], zero_division=0))
    
    print("\n--- Per-Distance Meta-Model Performance Breakdown ---")
    common_distances = sorted(list(set(all_distances)))
    for distance in common_distances:
        print(f"\n--- Distance: {distance} ft ---"); mask = all_distances == distance
        if np.sum(mask) == 0: continue
        print(classification_report(true_labels[mask], predictions[mask], target_names=['Normal', 'Anomaly'], labels=[0, 1], zero_division=0))
