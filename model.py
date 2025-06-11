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
EPOCHS = 40
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
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.startswith(prefix) and filename.endswith('.csv'):
                file_path = os.path.join(root, filename)
                try:
                    df = pd.read_csv(file_path)
                    all_dfs.append(df)
                except Exception as e:
                    print(f"  -> Could not read {filename}. Error: {e}")
    if not all_dfs:
        return None
    return pd.concat(all_dfs, ignore_index=True)

# --- 2. Preprocessing & Sequencing ---
def create_sequences(data, sequence_length):
    """Converts flat data into sequences."""
    sequences = []
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
        
        # Encoder: Compresses the sequence
        self.encoder = nn.Sequential(
            nn.LSTM(input_size, hidden_size, batch_first=True),
            nn.ReLU(),
            nn.LSTM(hidden_size, latent_size, batch_first=True),
            nn.ReLU()
        )
        
        # Decoder: Reconstructs the sequence from the compressed representation
        self.decoder = nn.Sequential(
            nn.LSTM(latent_size, hidden_size, batch_first=True),
            nn.ReLU(),
            nn.LSTM(hidden_size, input_size, batch_first=True),
        )

    def forward(self, x):
        # The output of the encoder is the last hidden state
        _, (hidden_state, _) = self.encoder(x)
        
        # We use this hidden state as the input for the decoder.
        # It needs to be repeated for each time step of the output sequence.
        latent_repeated = hidden_state.permute(1, 0, 2).repeat(1, x.size(1), 1)
        
        reconstructed = self.decoder(latent_repeated)
        return reconstructed

# --- Main Execution ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ONLY normal data for training
    normal_data = load_specific_data(DATA_DIR, 'no_jammer')
    if normal_data is None:
        print("ERROR: No 'no_jammer' files found. Cannot train the autoencoder.")
        exit()
        
    # Preprocess the normal data
    # Focus on key performance indicators, especially CQI
    features = ['dl_cqi', 'dl_brate', 'ul_rsrp', 'ul_mcs', 'dl_ok', 'dl_nok']
    
    # We don't need the UE identifier for this approach, simplifying the model
    X_normal = normal_data[features].astype(float)
    
    scaler = MinMaxScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    
    # Create training and validation sets from the normal data
    train_seq, val_seq = train_test_split(X_normal_scaled, test_size=0.2, random_state=42)
    
    train_sequences = create_sequences(train_seq, SEQUENCE_LENGTH)
    val_sequences = create_sequences(val_seq, SEQUENCE_LENGTH)

    train_dataset = SequenceDataset(train_sequences)
    val_dataset = SequenceDataset(val_sequences)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model, Loss, and Optimizer
    input_size = len(features)
    model = LSTMAutoencoder(input_size=input_size, sequence_len=SEQUENCE_LENGTH).to(device)
    criterion = nn.MSELoss() # Mean Squared Error is the reconstruction loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\n--- Model Architecture ---")
    print(model)
    
    # --- 4. Training Loop (only on normal data) ---
    print("\n--- Training Autoencoder on NORMAL data only ---")
    for epoch in range(EPOCHS):
        model.train()
        for seq, _ in train_loader:
            seq = seq.to(device)
            
            # Forward pass
            reconstructed = model(seq)
            loss = criterion(reconstructed, seq)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

    # --- 5. Determine Anomaly Threshold ---
    print("\n--- Calculating anomaly threshold from validation data ---")
    model.eval()
    losses = []
    with torch.no_grad():
        for seq, _ in val_loader:
            seq = seq.to(device)
            reconstructed = model(seq)
            loss = criterion(reconstructed, seq)
            losses.append(loss.item())
    
    # Set threshold to be mean + 3 standard deviations of the normal loss
    reconstruction_threshold = np.mean(losses) + 3 * np.std(losses)
    print(f"Calculated Reconstruction Error Threshold: {reconstruction_threshold:.6f}")

    # --- 6. Evaluate on ALL data (Normal and Jammed) ---
    print("\n--- Evaluating model on the full dataset ---")
    jammed_data = load_specific_data(DATA_DIR, 'jammer')
    
    # Create a full test set
    all_normal_sequences = create_sequences(X_normal_scaled, SEQUENCE_LENGTH)
    
    if jammed_data is not None:
        X_jammed = jammed_data[features].astype(float)
        X_jammed_scaled = scaler.transform(X_jammed) # Use the SAME scaler
        all_jammed_sequences = create_sequences(X_jammed_scaled, SEQUENCE_LENGTH)
        
        # Combine into a test set with labels
        test_sequences = np.concatenate([all_normal_sequences, all_jammed_sequences])
        true_labels = np.concatenate([np.zeros(len(all_normal_sequences)), np.ones(len(all_jammed_sequences))])
    else:
        print("Warning: No 'jammer' files found for final evaluation.")
        test_sequences = all_normal_sequences
        true_labels = np.zeros(len(all_normal_sequences))
        
    test_dataset = SequenceDataset(test_sequences)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    predictions = []
    model.eval()
    with torch.no_grad():
        for seq, _ in test_loader:
            seq = seq.to(device)
            reconstructed = model(seq)
            loss = torch.mean((reconstructed - seq) ** 2, dim=[1, 2])
            
            # Anomaly if loss > threshold
            preds = (loss > reconstruction_threshold).cpu().numpy().astype(int)
            predictions.extend(preds)

    print("\n--- Final Performance ---")
    print(classification_report(true_labels, predictions, target_names=['Normal (No Jamming)', 'Anomaly (Jamming)']))
    
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Anomaly Detection Confusion Matrix')
    plt.show()

    # --- 7. Save Model, Scaler, and Threshold ---
    print(f"\nSaving model to {MODEL_PATH}")
    torch.save(model.state_dict(), MODEL_PATH)
    
    print(f"Saving scaler to {SCALER_PATH}")
    joblib.dump(scaler, SCALER_PATH)

    print(f"Saving threshold to {THRESHOLD_PATH}")
    with open(THRESHOLD_PATH, 'w') as f:
        f.write(str(reconstruction_threshold))

    print("\nProcess complete.")
