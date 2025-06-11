import os
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json

def load_specific_data(root_dir, prefix):
    all_dfs = []
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
                except Exception as e:
                    print(f"  -> Could not read or process {filename}. Error: {e}")
    if not found_files: return None
    if not all_dfs: return None
    return pd.concat(all_dfs, ignore_index=True)

def create_sequences(data, sequence_length):
    sequences = []
    if len(data) <= sequence_length: return np.array(sequences)
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
    def __init__(self, input_size, hidden_size=64, latent_size=16, dropout_rate=0.2):
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

def plot_training_history(train_losses, val_losses, trial_number=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    title = 'Model Training History'
    if trial_number is not None:
        title += f" (Trial {trial_number})"
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    filename = "training_history.png" if trial_number is None else f"tuning_trial_{trial_number}_history.png"
    plt.savefig(filename)
    plt.close()

def run_training_and_evaluation(params, save_final_model=False, trial_number=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if trial_number is not None:
        print(f"\n--- Starting Trial #{trial_number} ---")
        print(f"Params: {params}")
        print(f"Device: {device}")

    DATA_DIR = params['data_dir']
    SEQUENCE_LENGTH = params['sequence_length']
    BATCH_SIZE = params['batch_size']
    EPOCHS = params['epochs']
    LEARNING_RATE = params['learning_rate']
    HIDDEN_SIZE = params['hidden_size']
    LATENT_SIZE = params['latent_size']
    DROPOUT_RATE = params['dropout_rate']
    THRESHOLD_PERCENTILE = params['threshold_percentile']
    EARLY_STOPPING_PATIENCE = params['early_stopping_patience']

    normal_data = load_specific_data(DATA_DIR, 'no_jammer')
    if normal_data is None or normal_data.empty: return 0.0

    features = ['dl_cqi', 'dl_brate', 'ul_rsrp', 'ul_mcs', 'dl_ok', 'dl_nok']
    missing_cols = [col for col in features if col not in normal_data.columns]
    if missing_cols: return 0.0

    for col in features:
        normal_data[col] = pd.to_numeric(normal_data[col], errors='coerce')
    normal_data.dropna(subset=features, inplace=True)

    if len(normal_data) < (SEQUENCE_LENGTH * 2 + 10): return 0.0

    scaler = MinMaxScaler()
    X_normal_scaled = scaler.fit_transform(normal_data[features].astype(float))
    train_data_scaled, val_data_scaled = train_test_split(X_normal_scaled, test_size=0.2, random_state=42)
    train_sequences = create_sequences(train_data_scaled, SEQUENCE_LENGTH)
    val_sequences = create_sequences(val_data_scaled, SEQUENCE_LENGTH)

    if len(train_sequences) == 0 or len(val_sequences) == 0: return 0.0

    train_loader = DataLoader(SequenceDataset(train_sequences), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SequenceDataset(val_sequences), batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMAutoencoder(input_size=len(features), hidden_size=HIDDEN_SIZE, latent_size=LATENT_SIZE, dropout_rate=DROPOUT_RATE).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # CORRECTED LINE: The 'verbose' argument has been removed.
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

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

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
        
        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            if save_final_model:
                torch.save(model.state_dict(), "best_model_checkpoint.pth")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if save_final_model:
        plot_training_history(train_loss_history, val_loss_history)
        model.load_state_dict(torch.load("best_model_checkpoint.pth"))

    model.eval()
    val_losses_per_sequence = []
    with torch.no_grad():
        for seq_val_batch, _ in val_loader:
            seq_val_batch = seq_val_batch.to(device)
            reconstructed_val = model(seq_val_batch)
            loss_per_item = torch.mean((reconstructed_val - seq_val_batch) ** 2, dim=[1, 2])
            val_losses_per_sequence.extend(loss_per_item.cpu().numpy())
    if not val_losses_per_sequence: return 0.0
    reconstruction_threshold = np.percentile(val_losses_per_sequence, THRESHOLD_PERCENTILE)

    jammed_data = load_specific_data(DATA_DIR, 'jammer')
    if jammed_data is None: return 0.0
    for col in features:
        jammed_data[col] = pd.to_numeric(jammed_data[col], errors='coerce')
    jammed_data.dropna(subset=features, inplace=True)
    if jammed_data.empty: return 0.0

    X_jammed_scaled = scaler.transform(jammed_data[features].astype(float))
    all_normal_sequences = create_sequences(X_normal_scaled, SEQUENCE_LENGTH)
    all_jammed_sequences = create_sequences(X_jammed_scaled, SEQUENCE_LENGTH)

    if len(all_normal_sequences) == 0 or len(all_jammed_sequences) == 0:
        return 0.0

    test_sequences = np.concatenate([all_normal_sequences, all_jammed_sequences])
    true_labels = np.concatenate([np.zeros(len(all_normal_sequences)), np.ones(len(all_jammed_sequences))])
    
    test_dataset = SequenceDataset(test_sequences)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for seq_test_batch, _ in test_loader:
            seq_test_batch = seq_test_batch.to(device)
            reconstructed_test = model(seq_test_batch)
            loss_test_per_item = torch.mean((reconstructed_test - seq_test_batch) ** 2, dim=[1, 2])
            preds = (loss_test_per_item > reconstruction_threshold).cpu().numpy().astype(int)
            predictions.extend(preds)

    report_dict = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
    anomaly_f1_score = report_dict.get('1', {}).get('f1-score', 0.0)
    print(f"Trial #{trial_number} finished with Anomaly F1-Score: {anomaly_f1_score:.4f}")

    if save_final_model:
        print("\n--- Final Performance Report ---")
        print(classification_report(true_labels, predictions, target_names=['Normal (0)', 'Anomaly (1)'], zero_division=0))
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Normal', 'Predicted Anomaly'], yticklabels=['Actual Normal', 'Actual Anomaly'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Final Model Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        print("Confusion matrix saved to confusion_matrix.png")
        os.rename("best_model_checkpoint.pth", "anomaly_autoencoder.pth")
        joblib.dump(scaler, "anomaly_scaler.pkl")
        params['reconstruction_threshold'] = reconstruction_threshold
        with open("best_params.json", 'w') as f:
            json.dump(params, f, indent=4)
        print("Artifacts saved.")

    return anomaly_f1_score

if __name__ == "__main__":
    default_params = {
        'data_dir': "srsran_csv_output",
        'sequence_length': 10,
        'batch_size': 64,
        'epochs': 300, 
        'early_stopping_patience': 15,
        'learning_rate': 0.001,
        'hidden_size': 64,
        'latent_size': 16,
        'dropout_rate': 0.2,
        'threshold_percentile': 95 
    }
    run_training_and_evaluation(default_params, save_final_model=True)