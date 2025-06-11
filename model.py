import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau # For learning rate scheduling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_DIR = "srsran_csv_output"
SEQUENCE_LENGTH = 10
BATCH_SIZE = 64
EPOCHS = 30000 # Increased epochs, will use early stopping
LEARNING_RATE = 0.001
MODEL_PATH = "anomaly_autoencoder_pytorch_v2.pth"
SCALER_PATH = "anomaly_scaler_pytorch_v2.pkl"
THRESHOLD_PATH = "anomaly_threshold_v2.txt"
DROPOUT_RATE = 0.2 # Added dropout
THRESHOLD_PERCENTILE = 80 # Changed threshold strategy
EARLY_STOPPING_PATIENCE = 10 # For early stopping

# --- 1. Data Loading ---
def load_specific_data(root_dir, prefix):
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
    if not found_files:
        print(f"  -> No non-empty files matching '{prefix}*.csv' found or successfully read.")
        return None
    if not all_dfs:
        print(f"  -> All matching files for '{prefix}*.csv' were empty.")
        return None
    return pd.concat(all_dfs, ignore_index=True)

# --- 2. Preprocessing & Sequencing ---
def create_sequences(data, sequence_length):
    sequences = []
    if len(data) <= sequence_length:
        print(f"Warning: Data length ({len(data)}) is less than or equal to sequence length ({sequence_length}). No sequences will be created.")
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

# --- 3. The LSTM Autoencoder Model ---
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, sequence_len, hidden_size=64, latent_size=16, dropout_rate=0.2):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder layers
        self.encoder_lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.encoder_dropout1 = nn.Dropout(dropout_rate) # Explicit dropout layer
        self.encoder_relu1 = nn.ReLU()
        
        self.encoder_lstm2 = nn.LSTM(hidden_size, latent_size, batch_first=True)
        self.encoder_dropout2 = nn.Dropout(dropout_rate) # Explicit dropout layer
        self.encoder_relu2 = nn.ReLU()
        
        # Decoder layers
        self.decoder_lstm1 = nn.LSTM(latent_size, hidden_size, batch_first=True)
        self.decoder_dropout3 = nn.Dropout(dropout_rate) # Explicit dropout layer
        self.decoder_relu1 = nn.ReLU()

        self.decoder_lstm2 = nn.LSTM(hidden_size, input_size, batch_first=True)
        # No dropout after the final output layer is common.

    def forward(self, x):
        # --- Encoder ---
        out_lstm1, _ = self.encoder_lstm1(x)
        out_dropout1 = self.encoder_dropout1(out_lstm1) # Apply dropout
        out_relu1 = self.encoder_relu1(out_dropout1)
        
        out_lstm2, (h_n_encoder, c_n_encoder) = self.encoder_lstm2(out_relu1)
        out_dropout2 = self.encoder_dropout2(out_lstm2) # Apply dropout
        encoded_sequence_activated = self.encoder_relu2(out_dropout2)
        
        latent_vector = encoded_sequence_activated[:, -1, :]
        
        # --- Decoder ---
        decoder_input_sequence = latent_vector.unsqueeze(1).repeat(1, x.size(1), 1)
        
        out_lstm3, _ = self.decoder_lstm1(decoder_input_sequence)
        out_dropout3 = self.decoder_dropout3(out_lstm3) # Apply dropout
        out_relu3 = self.decoder_relu1(out_dropout3)
        
        reconstructed_seq, _ = self.decoder_lstm2(out_relu3)
        
        return reconstructed_seq

# --- Utility Function to plot training history ---
def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_history.png")
    # plt.show()
    print("Training history plot saved to training_history.png")

# --- Main Execution ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    normal_data = load_specific_data(DATA_DIR, 'no_jammer')
    if normal_data is None or normal_data.empty:
        print("ERROR: No 'no_jammer' data found or loaded. Cannot train the autoencoder.")
        exit()
        
    features = ['dl_cqi']
    
    print("\n--- Cleaning normal_data ---")
    missing_cols = [col for col in features if col not in normal_data.columns]
    if missing_cols:
        print(f"ERROR: The following feature columns are missing in normal_data: {missing_cols}")
        exit()

    print("Original normal_data shape:", normal_data.shape)
    for col in features:
        if normal_data[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(normal_data[col]):
            if normal_data[col].dtype == 'object':
                print(f"  Converting column '{col}' in normal_data to numeric...")
                problematic_values = [val for val in normal_data[col].unique() if not isinstance(val, (int, float)) and not str(val).replace('.', '', 1).isdigit()]
                if problematic_values:
                    print(f"    Found non-numeric-like unique values in '{col}': {list(set(str(v) for v in problematic_values))[:10]}")
            else:
                print(f"  Column '{col}' is not object but also not strictly numeric ({normal_data[col].dtype}). Attempting conversion.")
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

    X_normal = normal_data[features].astype(float)
    
    scaler = MinMaxScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)
    
    train_data_scaled, val_data_scaled = train_test_split(X_normal_scaled, test_size=0.2, random_state=42)
    
    train_sequences = create_sequences(train_data_scaled, SEQUENCE_LENGTH)
    val_sequences = create_sequences(val_data_scaled, SEQUENCE_LENGTH)

    if len(train_sequences) == 0 or len(val_sequences) == 0:
        print("ERROR: Not enough data to create training or validation sequences after cleaning and splitting.")
        exit()

    train_dataset = SequenceDataset(train_sequences)
    val_dataset = SequenceDataset(val_sequences)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    input_size = len(features)
    model = LSTMAutoencoder(input_size=input_size, sequence_len=SEQUENCE_LENGTH, dropout_rate=DROPOUT_RATE).to(device)
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5) # LR scheduler

    print("\n--- Model Architecture ---")
    print(model)
    
    print("\n--- Training Autoencoder on NORMAL data only ---")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_loss_history = []
    val_loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        batch_train_losses = []
        for seq_batch, _ in train_loader:
            seq_batch = seq_batch.to(device)
            reconstructed = model(seq_batch)
            loss = criterion(reconstructed, seq_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())
        
        epoch_train_loss = np.mean(batch_train_losses)
        train_loss_history.append(epoch_train_loss)

        # Validation phase
        model.eval()
        batch_val_losses = []
        with torch.no_grad():
            for seq_val_batch, _ in val_loader:
                seq_val_batch = seq_val_batch.to(device)
                reconstructed_val = model(seq_val_batch)
                val_loss = criterion(reconstructed_val, seq_val_batch)
                batch_val_losses.append(val_loss.item())
        
        epoch_val_loss = np.mean(batch_val_losses)
        val_loss_history.append(epoch_val_loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
        
        scheduler.step(epoch_val_loss) # Step the scheduler

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), MODEL_PATH) 
            print(f"  Validation loss improved to {best_val_loss:.6f}. Model saved to {MODEL_PATH}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
            break
    
    plot_training_history(train_loss_history, val_loss_history)
    
    # Load the best model for threshold calculation and evaluation
    print(f"\nLoading best model from {MODEL_PATH} for threshold calculation and evaluation.")
    model.load_state_dict(torch.load(MODEL_PATH))

    print("\n--- Calculating anomaly threshold from validation data (per-sequence error) ---")
    model.eval()
    val_losses_per_sequence = []
    with torch.no_grad():
        for seq_val_batch, _ in val_loader: # Using the same val_loader
            seq_val_batch = seq_val_batch.to(device)
            reconstructed_val = model(seq_val_batch)
            loss_per_item = torch.mean((reconstructed_val - seq_val_batch) ** 2, dim=[1, 2])
            val_losses_per_sequence.extend(loss_per_item.cpu().numpy())
    
    if not val_losses_per_sequence:
        print("ERROR: No validation losses calculated. Cannot determine threshold.")
        exit()
        
    reconstruction_threshold = np.percentile(val_losses_per_sequence, THRESHOLD_PERCENTILE)
    print(f"Calculated Reconstruction Error Threshold ({THRESHOLD_PERCENTILE}th percentile): {reconstruction_threshold:.6f}")

    # --- Evaluation ---
    print("\n--- Evaluating model on the full dataset ---")
    jammed_data = load_specific_data(DATA_DIR, 'jammer')
    
    all_normal_sequences = create_sequences(X_normal_scaled, SEQUENCE_LENGTH)
    
    if jammed_data is not None and not jammed_data.empty:
        print("\n--- Cleaning jammed_data ---")
        missing_cols_jammed = [col for col in features if col not in jammed_data.columns]
        if missing_cols_jammed:
            print(f"ERROR: Features missing in jammed_data: {missing_cols_jammed}")
            jammed_data = None
        else:
            print("Original jammed_data shape:", jammed_data.shape)
            for col in features:
                if jammed_data[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(jammed_data[col]):
                    if jammed_data[col].dtype == 'object':
                        print(f"  Converting column '{col}' in jammed_data to numeric...")
                        problematic_values_jammed = [val for val in jammed_data[col].unique() if not isinstance(val, (int, float)) and not str(val).replace('.', '', 1).isdigit()]
                        if problematic_values_jammed:
                             print(f"    Found non-numeric-like unique values: {list(set(str(v) for v in problematic_values_jammed))[:10]}")
                    else:
                         print(f"  Column '{col}' is not object but also not strictly numeric ({jammed_data[col].dtype}). Attempting conversion.")
                    jammed_data[col] = pd.to_numeric(jammed_data[col], errors='coerce')

            nan_rows_jammed = jammed_data[features].isnull().any(axis=1)
            if nan_rows_jammed.sum() > 0:
                print(f"  Found {nan_rows_jammed.sum()} rows with non-convertible values (now NaN) in jammed_data.")
                jammed_data.dropna(subset=features, inplace=True)
                print(f"  Number of rows in jammed_data after dropping NaNs: {len(jammed_data)}")
        
        if jammed_data is None or jammed_data.empty:
            print("Warning: jammed_data is empty or unusable after cleaning.")
            all_jammed_sequences = np.array([])
        else:
            X_jammed = jammed_data[features].astype(float)
            X_jammed_scaled = scaler.transform(X_jammed) 
            all_jammed_sequences = create_sequences(X_jammed_scaled, SEQUENCE_LENGTH)
    else:
        print("Warning: No 'jammer' files found or loaded for final evaluation.")
        all_jammed_sequences = np.array([])
        
    if len(all_normal_sequences) == 0:
        print("ERROR: No normal sequences available for testing.")
        exit()

    test_sequences_list = [all_normal_sequences]
    true_labels_list = [np.zeros(len(all_normal_sequences))]

    if len(all_jammed_sequences) > 0:
        test_sequences_list.append(all_jammed_sequences)
        true_labels_list.append(np.ones(len(all_jammed_sequences)))
        print(f"Testing with {len(all_normal_sequences)} normal sequences and {len(all_jammed_sequences)} jammed sequences.")
    else:
        print(f"Testing with {len(all_normal_sequences)} normal sequences only (no valid jammed sequences).")

    test_sequences_np = np.concatenate(test_sequences_list) if test_sequences_list else np.array([])
    true_labels = np.concatenate(true_labels_list) if true_labels_list else np.array([])
        
    if len(test_sequences_np) == 0:
        print("ERROR: No sequences available for testing.")
        exit()

    test_dataset = SequenceDataset(test_sequences_np)
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
        plt.savefig("confusion_matrix_v2.png")
        print("Confusion matrix saved to confusion_matrix_v2.png")

    # Visualize reconstruction errors (Added from previous suggestion)
    normal_test_losses = []
    jammed_test_losses = []
    model.eval()
    with torch.no_grad():
        if len(all_normal_sequences) > 0:
            normal_test_dataset = SequenceDataset(all_normal_sequences)
            normal_test_loader = DataLoader(normal_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            for seq_batch, _ in normal_test_loader:
                seq_batch = seq_batch.to(device)
                reconstructed = model(seq_batch)
                loss_per_item = torch.mean((reconstructed - seq_batch) ** 2, dim=[1, 2])
                normal_test_losses.extend(loss_per_item.cpu().numpy())

        if len(all_jammed_sequences) > 0:
            jammed_test_dataset = SequenceDataset(all_jammed_sequences)
            jammed_test_loader = DataLoader(jammed_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
            for seq_batch, _ in jammed_test_loader:
                seq_batch = seq_batch.to(device)
                reconstructed = model(seq_batch)
                loss_per_item = torch.mean((reconstructed - seq_batch) ** 2, dim=[1, 2])
                jammed_test_losses.extend(loss_per_item.cpu().numpy())

    plt.figure(figsize=(12, 7))
    if val_losses_per_sequence:
        sns.histplot(val_losses_per_sequence, color="blue", label=f"Normal Val Errors ({len(val_losses_per_sequence)} samples)", kde=True, stat="density", element="step", alpha=0.6)
    if normal_test_losses:
        sns.histplot(normal_test_losses, color="green", label=f"Normal Test Errors ({len(normal_test_losses)} samples)", kde=True, stat="density", element="step", alpha=0.6)
    if jammed_test_losses:
       sns.histplot(jammed_test_losses, color="red", label=f"Jammed Test Errors ({len(jammed_test_losses)} samples)", kde=True, stat="density", element="step", alpha=0.6)
    plt.axvline(reconstruction_threshold, color="black", linestyle="--", label=f"Threshold ({THRESHOLD_PERCENTILE}th Perc. = {reconstruction_threshold:.4f})")
    plt.title("Reconstruction Error Distributions (Test Data vs. Validation Normal)")
    plt.xlabel("Mean Squared Error per Sequence")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.savefig("reconstruction_error_distributions_v2.png")
    print("Reconstruction error distributions plot saved to reconstruction_error_distributions_v2.png")

    print(f"\nSaving best model to {MODEL_PATH}") # Already saved during training
    # torch.save(model.state_dict(), MODEL_PATH) # Redundant if early stopping saves best
    
    print(f"Saving scaler to {SCALER_PATH}")
    joblib.dump(scaler, SCALER_PATH)

    print(f"Saving threshold to {THRESHOLD_PATH}")
    with open(THRESHOLD_PATH, 'w') as f:
        f.write(str(reconstruction_threshold))

    print("\nProcess complete.")