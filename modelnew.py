import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_DIR = "srsran_csv_output"
UE_IDENTIFIER = "ue2"
SEQUENCE_LENGTH = 10
BATCH_SIZE = 64
EPOCHS = 30000
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2
THRESHOLD_PERCENTILE = 95 # Set higher to be more sensitive to anomalies
EARLY_STOPPING_PATIENCE = 10
# Define the new, engineered feature set. These will be created from dl_cqi.
FEATURES = ['dl_cqi', 'cqi_diff', 'cqi_rolling_std']

# --- 1. Data Loading ---
def load_all_data(root_dir, prefix, ue_filter):
    """
    Loads and concatenates ALL CSV files that match a prefix and contain a UE identifier.
    """
    all_dfs = []
    print(f"Searching for all '{prefix}*.csv' files for '{ue_filter}' in '{root_dir}'...")
    for root, _, files in os.walk(root_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            if filename.startswith(prefix) and filename.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    if not df.empty and 'ue_identifier' in df.columns and ue_filter in df['ue_identifier'].unique():
                        all_dfs.append(df[df['ue_identifier'] == ue_filter])
                except Exception as e:
                    print(f"  -> Could not read or process {filename}. Error: {e}")

    if not all_dfs:
        print(f"  -> No non-empty files matching '{prefix}*.csv' with '{ue_filter}' data found.")
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"  -> Loaded {len(combined_df)} total rows from matching files.")
    return combined_df

# --- 2. Preprocessing, Model, and Utilities ---
def create_sequences(data, sequence_length):
    sequences = []
    if len(data) <= sequence_length:
        return np.array(sequences)
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:(i + sequence_length)])
    return np.array(sequences)

class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.sequences[idx]

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, sequence_len, hidden_size=64, latent_size=16, dropout_rate=0.2):
        super(LSTMAutoencoder, self).__init__()
        self.encoder_lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.encoder_dropout1 = nn.Dropout(dropout_rate)
        self.encoder_relu1 = nn.ReLU()
        self.encoder_lstm2 = nn.LSTM(hidden_size, latent_size, batch_first=True)
        self.encoder_dropout2 = nn.Dropout(dropout_rate)
        self.encoder_relu2 = nn.ReLU()
        self.decoder_lstm1 = nn.LSTM(latent_size, hidden_size, batch_first=True)
        self.decoder_dropout3 = nn.Dropout(dropout_rate)
        self.decoder_relu1 = nn.ReLU()
        self.decoder_lstm2 = nn.LSTM(hidden_size, input_size, batch_first=True)

    def forward(self, x):
        out_lstm1, _ = self.encoder_lstm1(x)
        out_dropout1 = self.encoder_dropout1(out_lstm1)
        out_relu1 = self.encoder_relu1(out_dropout1)
        out_lstm2, (h_n_encoder, c_n_encoder) = self.encoder_lstm2(out_relu1)
        out_dropout2 = self.encoder_dropout2(out_lstm2)
        encoded_sequence_activated = self.encoder_relu2(out_dropout2)
        latent_vector = encoded_sequence_activated[:, -1, :]
        decoder_input_sequence = latent_vector.unsqueeze(1).repeat(1, x.size(1), 1)
        out_lstm3, _ = self.decoder_lstm1(decoder_input_sequence)
        out_dropout3 = self.decoder_dropout3(out_lstm3)
        out_relu3 = self.decoder_relu1(out_dropout3)
        reconstructed_seq, _ = self.decoder_lstm2(out_relu3)
        return reconstructed_seq

def plot_training_history(train_losses, val_losses, distance):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Model Training History (Distance: {distance} ft)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    filename = f"training_history_{distance}ft.png"
    plt.savefig(filename)
    plt.close()
    print(f"Training history plot saved to {filename}")

# --- 3. Main Training and Evaluation Loop ---
def train_and_evaluate_for_distance(distance, full_normal_df, full_jammed_df):
    """
    This function encapsulates the entire process for a single distance.
    This version engineers new features to capture signal volatility.
    """
    print("\n" + "="*80)
    print(f"=== PROCESSING DISTANCE: {distance} ft (with Feature Engineering) ===")
    print("="*80 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Feature Engineering Step ---
    print("--- Engineering new features (diff, rolling_std) ---")
    
    # Process Normal Data
    normal_data = full_normal_df[full_normal_df['ue2_distance_ft'] == distance].copy()
    normal_data['dl_cqi'] = pd.to_numeric(normal_data['dl_cqi'], errors='coerce')
    # IMPORTANT: Ensure data is in time order if possible.
    # If a timestamp column exists, sort by it first.
    normal_data.sort_index(inplace=True) # A simple sort to ensure order before diff
    normal_data['cqi_diff'] = normal_data['dl_cqi'].diff().fillna(0)
    normal_data['cqi_rolling_std'] = normal_data['dl_cqi'].rolling(window=5).std().fillna(0)
    normal_data.dropna(subset=FEATURES, inplace=True)

    # Process Jammed Data
    jammed_data = full_jammed_df[full_jammed_df['ue2_distance_ft'] == distance].copy()
    jammed_data['dl_cqi'] = pd.to_numeric(jammed_data['dl_cqi'], errors='coerce')
    jammed_data.sort_index(inplace=True) # A simple sort to ensure order before diff
    jammed_data['cqi_diff'] = jammed_data['dl_cqi'].diff().fillna(0)
    jammed_data['cqi_rolling_std'] = jammed_data['dl_cqi'].rolling(window=5).std().fillna(0)
    jammed_data.dropna(subset=FEATURES, inplace=True)
    
    if len(normal_data) < (SEQUENCE_LENGTH * 2):
        print(f"ERROR: Not enough normal data ({len(normal_data)} rows) after feature engineering. Skipping.")
        return None

    # --- Scaling and Sequencing ---
    X_normal = normal_data[FEATURES].astype(float)
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)

    train_data_scaled, val_data_scaled = train_test_split(X_normal_scaled, test_size=0.2, random_state=42)
    train_sequences = create_sequences(train_data_scaled, SEQUENCE_LENGTH)
    val_sequences = create_sequences(val_data_scaled, SEQUENCE_LENGTH)
    if len(train_sequences) == 0 or len(val_sequences) == 0:
        print(f"ERROR: Not enough data for train/val sequences. Skipping.")
        return None

    train_loader = DataLoader(SequenceDataset(train_sequences), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SequenceDataset(val_sequences), batch_size=BATCH_SIZE, shuffle=False)

    model_path = f"model_{distance}ft.pth"
    model = LSTMAutoencoder(input_size=len(FEATURES), sequence_len=SEQUENCE_LENGTH, dropout_rate=DROPOUT_RATE).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    print(f"--- Training model for {distance} ft on {len(FEATURES)} features ---")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_loss_history, val_loss_history = [], []

    for epoch in range(EPOCHS):
        model.train()
        epoch_batch_losses = []
        for seq, _ in train_loader:
            seq_on_device = seq.to(device)
            reconstructed = model(seq_on_device)
            loss = criterion(reconstructed, seq_on_device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_batch_losses.append(loss.item())
        epoch_train_loss = np.mean(epoch_batch_losses)
        train_loss_history.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            batch_val_losses = [criterion(model(seq.to(device)), seq.to(device)).item() for seq, _ in val_loader]
            epoch_val_loss = np.mean(batch_val_losses)
            val_loss_history.append(epoch_val_loss)

        if (epoch + 1) % 10 == 0:
             print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
        
        scheduler.step(epoch_val_loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss, epochs_no_improve = epoch_val_loss, 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    plot_training_history(train_loss_history, val_loss_history, distance)

    print(f"\nLoading best model from {model_path} and calculating threshold...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    val_losses_per_sequence = []
    with torch.no_grad():
        for seq_val_batch, _ in val_loader:
            reconstructed_val = model(seq_val_batch.to(device))
            loss_per_item = torch.mean((reconstructed_val - seq_val_batch.to(device)) ** 2, dim=[1, 2])
            val_losses_per_sequence.extend(loss_per_item.cpu().numpy())
    
    reconstruction_threshold = np.percentile(val_losses_per_sequence, THRESHOLD_PERCENTILE)
    print(f"Calculated Threshold ({THRESHOLD_PERCENTILE}th percentile) for {distance} ft: {reconstruction_threshold:.6f}")

    print("\n--- Evaluating model ---")
    all_normal_sequences = create_sequences(X_normal_scaled, SEQUENCE_LENGTH)
    
    X_jammed = jammed_data[FEATURES].astype(float)
    X_jammed_scaled = scaler.transform(X_jammed)
    all_jammed_sequences = create_sequences(X_jammed_scaled, SEQUENCE_LENGTH)
    
    if len(all_jammed_sequences) == 0:
        print(f"WARNING: No jammed sequences available for evaluation. Skipping.")
        return None

    test_sequences_np = np.concatenate([all_normal_sequences, all_jammed_sequences])
    true_labels = np.concatenate([np.zeros(len(all_normal_sequences)), np.ones(len(all_jammed_sequences))])

    test_loader = DataLoader(SequenceDataset(test_sequences_np), batch_size=BATCH_SIZE, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for seq_test_batch, _ in test_loader:
            reconstructed_test = model(seq_test_batch.to(device))
            loss_per_item = torch.mean((reconstructed_test - seq_test_batch.to(device)) ** 2, dim=[1, 2])
            preds = (loss_per_item > reconstruction_threshold).cpu().numpy().astype(int)
            predictions.extend(preds)

    print(f"\n--- Final Performance for {distance} ft ---")
    report = classification_report(true_labels, predictions,
                                   target_names=['Normal', 'Anomaly'],
                                   zero_division=0, output_dict=True)
    print(classification_report(true_labels, predictions, target_names=['Normal', 'Anomaly'], zero_division=0))
    
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Normal', 'Pred Anomaly'], yticklabels=['Actual Normal', 'Actual Anomaly'])
    plt.title(f'Confusion Matrix (Distance: {distance} ft)')
    plt.savefig(f"confusion_matrix_{distance}ft.png")
    plt.close()

    joblib.dump(scaler, f"scaler_{distance}ft.pkl")
    with open(f"threshold_{distance}ft.txt", 'w') as f:
        f.write(str(reconstruction_threshold))
    print(f"Saved model, scaler, and threshold for {distance} ft.")

    return report


if __name__ == "__main__":
    normal_df_all = load_all_data(DATA_DIR, 'no_jammer', UE_IDENTIFIER)
    jammed_df_all = load_all_data(DATA_DIR, 'jammer', UE_IDENTIFIER)

    if normal_df_all is None or jammed_df_all is None:
        print("\nERROR: Could not load both normal and jammed datasets. Exiting.")
        exit()

    print("\n--- Discovering common distances between normal and jammed datasets ---")
    for df in [normal_df_all, jammed_df_all]:
        df['ue2_distance_ft'] = pd.to_numeric(df['ue2_distance_ft'], errors='coerce')
        df.dropna(subset=['ue2_distance_ft'], inplace=True)
        df['ue2_distance_ft'] = df['ue2_distance_ft'].astype(int)

    normal_distances = set(normal_df_all['ue2_distance_ft'].unique())
    jammed_distances = set(jammed_df_all['ue2_distance_ft'].unique())
    common_distances = sorted(list(normal_distances.intersection(jammed_distances)))

    if not common_distances:
        print("\nFATAL ERROR: No common distances found between the 'no_jammer' and 'jammer' datasets.")
        print(f"  Normal distances found: {sorted(list(normal_distances))}")
        print(f"  Jammed distances found: {sorted(list(jammed_distances))}")
        exit()

    print(f"\nFound {len(common_distances)} common distance(s) to process: {common_distances} ft")

    all_results = {}
    for distance in common_distances:
        report = train_and_evaluate_for_distance(distance, normal_df_all, jammed_df_all)
        if report:
            all_results[distance] = report

    print("\n" + "#"*80)
    print("### FINAL SUMMARY OF RESULTS ACROSS ALL DISTANCES ###")
    print("#"*80 + "\n")

    for distance, report in all_results.items():
        f1_anomaly = report.get('Anomaly', {}).get('f1-score', 'N/A')
        f1_normal = report.get('Normal', {}).get('f1-score', 'N/A')
        accuracy = report.get('accuracy', 'N/A')
        
        print(f"Distance: {distance} ft")
        print(f"  - Accuracy: {accuracy:.3f}")
        print(f"  - F1-Score (Normal): {f1_normal:.3f}")
        print(f"  - F1-Score (Anomaly): {f1_anomaly:.3f}")
        print("-" * 30)

    print("\nProcess complete.")