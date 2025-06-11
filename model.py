import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_DIR = "srsran_csv_output"
SEQUENCE_LENGTH = 10
BATCH_SIZE = 64
EPOCHS = 40 # Consider increasing if loss is still decreasing
MODEL_PATH = "anomaly_autoencoder_pytorch.pth"
SCALER_PATH = "anomaly_scaler_pytorch.pkl"
THRESHOLD_PATH = "anomaly_threshold.txt"

# --- 1. Data Loading ---
def load_specific_data(root_dir, prefix):
    """
    Loads all data files starting with a specific prefix (e.g., 'no_jammer').
    """
    all_dfs = []
    print(f"Searching for '{prefix}*.csv' in '{root_dir}'...")
    found_files = False
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.startswith(prefix) and filename.endswith('.csv'):
                file_path = os.path.join(root, filename)
                try:
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        all_dfs.append(df)
                        found_files = True
                    else:
                        print(f"  -> Skipped empty file: {filename}")
                except Exception as e:
                    print(f"  -> Could not read or process {filename}. Error: {e}")
    if not found_files: # Changed condition to check if any files were successfully processed
        print(f"  -> No non-empty files matching '{prefix}*.csv' found or successfully read.")
        return None
    if not all_dfs: # If found_files was true but all_dfs is empty (e.g. all files were empty)
        print(f"  -> All matching files for '{prefix}*.csv' were empty.")
        return None
    return pd.concat(all_dfs, ignore_index=True)

# --- 2. Preprocessing & Sequencing ---
def create_sequences(data, sequence_length):
    """Converts flat data into sequences."""
    sequences = []
    if len(data) <= sequence_length: # Check if data is too short
        print(f"Warning: Data length ({len(data)}) is less than or equal to sequence length ({sequence_length}). No sequences will be created.")
        return np.array(sequences) # Return empty array
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:(i + sequence_length)])
    return np.array(sequences)

class SequenceDataset(Dataset):
    """Custom PyTorch Dataset for sequence data."""
    def __init__(self, sequences):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.sequences[idx] # For autoencoders, input and target are the same

# --- 3. The LSTM Autoencoder Model ---
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, sequence_len, hidden_size=64, latent_size=16):
        super(LSTMAutoencoder, self).__init__()
        # sequence_len is not strictly used in this layer definition but can be kept for info
        
        # Encoder layers
        self.encoder_lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.encoder_relu1 = nn.ReLU()
        self.encoder_lstm2 = nn.LSTM(hidden_size, latent_size, batch_first=True)
        self.encoder_relu2 = nn.ReLU() # This ReLU will act on the output sequence of encoder_lstm2
        
        # Decoder layers
        self.decoder_lstm1 = nn.LSTM(latent_size, hidden_size, batch_first=True)
        self.decoder_relu1 = nn.ReLU()
        self.decoder_lstm2 = nn.LSTM(hidden_size, input_size, batch_first=True)
        # No final ReLU in the decoder is common for reconstruction tasks to allow any value range.

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        
        # --- Encoder ---
        # Layer 1
        out_lstm1, _ = self.encoder_lstm1(x)  # LSTM returns (output_sequence, (h_n, c_n))
        out_relu1 = self.encoder_relu1(out_lstm1)
        
        # Layer 2
        # The output sequence from LSTM1 (after ReLU) is fed into LSTM2
        out_lstm2, (h_n_encoder, c_n_encoder) = self.encoder_lstm2(out_relu1) 
        # out_lstm2 shape: (batch, seq_len, latent_size)
        # h_n_encoder shape: (num_layers*num_directions, batch, latent_size), here (1, batch, latent_size)
        
        # Apply ReLU to the entire output sequence of the second LSTM
        encoded_sequence_activated = self.encoder_relu2(out_lstm2)
        # encoded_sequence_activated shape: (batch, seq_len, latent_size)

        # The latent representation is typically the output of the encoder at the last time step,
        # or the final hidden state. Using the last time step of the activated sequence:
        latent_vector = encoded_sequence_activated[:, -1, :] # Shape: (batch, latent_size)
        
        # --- Decoder ---
        # The decoder LSTM needs an input sequence. We repeat the latent_vector for seq_len times.
        # latent_vector shape: (batch, latent_size)
        # Unsqueeze to (batch, 1, latent_size) then repeat. x.size(1) is current_seq_len.
        decoder_input_sequence = latent_vector.unsqueeze(1).repeat(1, x.size(1), 1)
        # decoder_input_sequence shape: (batch, seq_len, latent_size)
        
        # Layer 1
        out_lstm3, _ = self.decoder_lstm1(decoder_input_sequence)
        out_relu3 = self.decoder_relu1(out_lstm3)
        
        # Layer 2
        # The output sequence from LSTM3 (after ReLU) is fed into LSTM4
        reconstructed_seq, _ = self.decoder_lstm2(out_relu3)
        # reconstructed_seq shape: (batch, seq_len, input_size)
        
        return reconstructed_seq

# --- Main Execution ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ONLY normal data for training
    normal_data = load_specific_data(DATA_DIR, 'no_jammer')
    if normal_data is None or normal_data.empty:
        print("ERROR: No 'no_jammer' data found or loaded. Cannot train the autoencoder.")
        exit()
        
    features = ['dl_cqi', 'dl_brate', 'ul_rsrp', 'ul_mcs', 'dl_ok', 'dl_nok']
    
    # --- Data Cleaning for normal_data ---
    print("\n--- Cleaning normal_data ---")
    # Ensure selected feature columns exist
    missing_cols = [col for col in features if col not in normal_data.columns]
    if missing_cols:
        print(f"ERROR: The following feature columns are missing in normal_data: {missing_cols}")
        exit()

    print("Original normal_data shape:", normal_data.shape)
    for col in features:
        if normal_data[col].dtype == 'object':
            print(f"  Converting column '{col}' in normal_data to numeric...")
            # Identify problematic string values before conversion for logging
            problematic_values = []
            unique_vals = normal_data[col].unique()
            for val in unique_vals:
                try:
                    float(val)
                except (ValueError, TypeError):
                    problematic_values.append(val)
            if problematic_values:
                print(f"    Found non-numeric-like unique values in '{col}': {list(problematic_values)[:10]} (showing up to 10)")
            normal_data[col] = pd.to_numeric(normal_data[col], errors='coerce')
            if normal_data[col].isnull().any():
                print(f"    NaNs introduced in '{col}' after pd.to_numeric.")
        elif not pd.api.types.is_numeric_dtype(normal_data[col]):
             print(f"  Column '{col}' is not object type but also not numeric ({normal_data[col].dtype}). Attempting conversion.")
             normal_data[col] = pd.to_numeric(normal_data[col], errors='coerce')
             if normal_data[col].isnull().any():
                print(f"    NaNs introduced in '{col}' after pd.to_numeric.")


    nan_rows_normal = normal_data[features].isnull().any(axis=1)
    if nan_rows_normal.sum() > 0:
        print(f"  Found {nan_rows_normal.sum()} rows with non-convertible values (now NaN) in feature columns of normal_data.")
        normal_data.dropna(subset=features, inplace=True)
        print(f"  Number of rows in normal_data after dropping NaNs: {len(normal_data)}")

    if normal_data.empty:
        print("ERROR: normal_data is empty after cleaning. Cannot proceed.")
        exit()

    X_normal = normal_data[features].astype(float) # Ensure float type
    
    scaler = MinMaxScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    
    train_data_scaled, val_data_scaled = train_test_split(X_normal_scaled, test_size=0.2, random_state=42)
    
    train_sequences = create_sequences(train_data_scaled, SEQUENCE_LENGTH)
    val_sequences = create_sequences(val_data_scaled, SEQUENCE_LENGTH)

    if len(train_sequences) == 0 or len(val_sequences) == 0:
        print("ERROR: Not enough data to create training or validation sequences after cleaning and splitting.")
        print(f"Training sequences: {len(train_sequences)}, Validation sequences: {len(val_sequences)}")
        exit()

    train_dataset = SequenceDataset(train_sequences)
    val_dataset = SequenceDataset(val_sequences)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    input_size = len(features)
    model = LSTMAutoencoder(input_size=input_size, sequence_len=SEQUENCE_LENGTH).to(device)
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\n--- Model Architecture ---")
    print(model)
    
    print("\n--- Training Autoencoder on NORMAL data only ---")
    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        for seq_batch, _ in train_loader:
            seq_batch = seq_batch.to(device)
            reconstructed = model(seq_batch)
            loss = criterion(reconstructed, seq_batch) # criterion is MSELoss(reduction='mean') by default
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Average Training Loss: {np.mean(batch_losses):.6f}")

    print("\n--- Calculating anomaly threshold from validation data (per-sequence error) ---")
    model.eval()
    val_losses_per_sequence = []
    with torch.no_grad():
        for seq_val_batch, _ in val_loader:
            seq_val_batch = seq_val_batch.to(device)
            reconstructed_val = model(seq_val_batch)
            # Calculate MSE loss per sequence in the batch
            loss_per_item = torch.mean((reconstructed_val - seq_val_batch) ** 2, dim=[1, 2]) # Shape: (batch_size)
            val_losses_per_sequence.extend(loss_per_item.cpu().numpy())
    
    if not val_losses_per_sequence:
        print("ERROR: No validation losses calculated. Cannot determine threshold. Check validation data and sequence creation.")
        exit()
        
    reconstruction_threshold = np.mean(val_losses_per_sequence) + 3 * np.std(val_losses_per_sequence)
    print(f"Calculated Reconstruction Error Threshold: {reconstruction_threshold:.6f}")

    print("\n--- Evaluating model on the full dataset ---")
    jammed_data = load_specific_data(DATA_DIR, 'jammer')
    
    all_normal_sequences = create_sequences(X_normal_scaled, SEQUENCE_LENGTH) # Recreate from all normal data for consistency
    
    if jammed_data is not None and not jammed_data.empty:
        print("\n--- Cleaning jammed_data ---")
        # Ensure selected feature columns exist
        missing_cols_jammed = [col for col in features if col not in jammed_data.columns]
        if missing_cols_jammed:
            print(f"ERROR: The following feature columns are missing in jammed_data: {missing_cols_jammed}")
            # Decide how to handle: skip jammed data evaluation or exit
            jammed_data = None # Skip evaluation with jammed data
        else:
            print("Original jammed_data shape:", jammed_data.shape)
            for col in features:
                if jammed_data[col].dtype == 'object':
                    print(f"  Converting column '{col}' in jammed_data to numeric...")
                    problematic_values_jammed = []
                    unique_vals_jammed = jammed_data[col].unique()
                    for val_j in unique_vals_jammed:
                        try:
                            float(val_j)
                        except (ValueError, TypeError):
                            problematic_values_jammed.append(val_j)
                    if problematic_values_jammed:
                        print(f"    Found non-numeric-like unique values: {list(problematic_values_jammed)[:10]}")
                    jammed_data[col] = pd.to_numeric(jammed_data[col], errors='coerce')
                elif not pd.api.types.is_numeric_dtype(jammed_data[col]):
                    print(f"  Column '{col}' is not object type but also not numeric ({jammed_data[col].dtype}). Attempting conversion.")
                    jammed_data[col] = pd.to_numeric(jammed_data[col], errors='coerce')

            nan_rows_jammed = jammed_data[features].isnull().any(axis=1)
            if nan_rows_jammed.sum() > 0:
                print(f"  Found {nan_rows_jammed.sum()} rows with non-convertible values (now NaN) in jammed_data.")
                jammed_data.dropna(subset=features, inplace=True)
                print(f"  Number of rows in jammed_data after dropping NaNs: {len(jammed_data)}")
        
        if jammed_data is None or jammed_data.empty:
            print("Warning: jammed_data is empty or became unusable after cleaning. Evaluation will only use normal data.")
            all_jammed_sequences = np.array([])
        else:
            X_jammed = jammed_data[features].astype(float)
            X_jammed_scaled = scaler.transform(X_jammed) 
            all_jammed_sequences = create_sequences(X_jammed_scaled, SEQUENCE_LENGTH)
    else:
        print("Warning: No 'jammer' files found or loaded for final evaluation.")
        all_jammed_sequences = np.array([])
        
    if len(all_normal_sequences) == 0:
        print("ERROR: No normal sequences available for testing. This should not happen if training was successful.")
        exit()

    test_sequences_list = [all_normal_sequences]
    true_labels_list = [np.zeros(len(all_normal_sequences))]

    if len(all_jammed_sequences) > 0:
        test_sequences_list.append(all_jammed_sequences)
        true_labels_list.append(np.ones(len(all_jammed_sequences)))
        print(f"Testing with {len(all_normal_sequences)} normal sequences and {len(all_jammed_sequences)} jammed sequences.")
    else:
        print(f"Testing with {len(all_normal_sequences)} normal sequences only (no valid jammed sequences).")

    test_sequences = np.concatenate(test_sequences_list)
    true_labels = np.concatenate(true_labels_list)
        
    if len(test_sequences) == 0:
        print("ERROR: No sequences available for testing.")
        exit()

    test_dataset = SequenceDataset(test_sequences)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    predictions = []
    model.eval()
    with torch.no_grad():
        for seq_test_batch, _ in test_loader:
            seq_test_batch = seq_test_batch.to(device)
            reconstructed_test = model(seq_test_batch)
            loss_test_per_item = torch.mean((reconstructed_test - seq_test_batch) ** 2, dim=[1, 2])
            preds = (loss_test_per_item > reconstruction_threshold).cpu().numpy().astype(int)
            predictions.extend(preds)

    if not predictions:
        print("Warning: No predictions were made. Check test data.")
    else:
        print("\n--- Final Performance ---")
        # Handle cases where only one class might be present in true_labels or predictions
        # (e.g. if no jammed data or all data classified as normal)
        report = classification_report(true_labels, predictions, 
                                       target_names=['Normal (No Jamming)', 'Anomaly (Jamming)'],
                                       zero_division=0)
        print(report)
        
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted Normal', 'Predicted Anomaly'], 
                    yticklabels=['Actual Normal', 'Actual Anomaly'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Anomaly Detection Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        print("Confusion matrix saved to confusion_matrix.png")
        # plt.show() # Uncomment if you want to display it interactively

    print(f"\nSaving model to {MODEL_PATH}")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saving scaler to {SCALER_PATH}")
    joblib.dump(scaler, SCALER_PATH)

    print(f"Saving threshold to {THRESHOLD_PATH}")
    with open(THRESHOLD_PATH, 'w') as f:
        f.write(str(reconstruction_threshold))

    print("\nProcess complete.")