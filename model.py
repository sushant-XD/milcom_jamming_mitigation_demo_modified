# model.py (Corrected with create_sequences restored)
import os
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

# --- Utility Functions ---
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

# --- Model Definition ---
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
        out, _ = self.encoder_lstm1(x)
        out = self.encoder_dropout1(out)
        out = self.encoder_relu1(out)
        out, _ = self.encoder_lstm2(out)
        out = self.encoder_dropout2(out)
        out = self.encoder_relu2(out)
        latent_vector = out[:, -1, :]
        decoder_input = latent_vector.unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder_lstm1(decoder_input)
        out = self.decoder_dropout3(out)
        out = self.decoder_relu1(out)
        reconstructed_seq, _ = self.decoder_lstm2(out)
        return reconstructed_seq

def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch'); plt.ylabel('Loss (MSE)')
    plt.legend(); plt.grid(True)
    plt.savefig("training_history.png")
    plt.close()

# --- Main Training and Evaluation Function ---
def run_training_and_evaluation(params, prepared_data, save_final_model=False, trial_number=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if trial_number is not None:
        print(f"\n--- Starting Trial #{trial_number} on {device} ---")

    BATCH_SIZE = params['batch_size']
    EPOCHS = params['epochs']
    LEARNING_RATE = params['learning_rate']
    HIDDEN_SIZE = params['hidden_size']
    LATENT_SIZE = params['latent_size']
    DROPOUT_RATE = params['dropout_rate']
    THRESHOLD_PERCENTILE = params['threshold_percentile']
    EARLY_STOPPING_PATIENCE = params['early_stopping_patience']
    
    NUM_WORKERS = min(os.cpu_count(), 4) if device.type == 'cuda' else 0
    PIN_MEMORY = True if device.type == 'cuda' else False

    train_sequences = prepared_data['train']
    val_sequences = prepared_data['validation']
    
    if len(train_sequences) == 0 or len(val_sequences) == 0: return 0.0

    train_loader = DataLoader(SequenceDataset(train_sequences), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(SequenceDataset(val_sequences), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    input_size = train_sequences.shape[2]
    model = LSTMAutoencoder(input_size, HIDDEN_SIZE, LATENT_SIZE, DROPOUT_RATE).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_loss_history, val_loss_history = [], []
    
    for epoch in range(EPOCHS):
        model.train()
        for seq_batch, _ in train_loader:
            seq_batch = seq_batch.to(device)
            reconstructed = model(seq_batch)
            loss = criterion(reconstructed, seq_batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        model.eval()
        batch_val_losses = []
        with torch.no_grad():
            for seq_val_batch, _ in val_loader:
                reconstructed_val = model(seq_val_batch.to(device))
                val_loss = criterion(reconstructed_val, seq_val_batch.to(device))
                batch_val_losses.append(val_loss.item())
        epoch_val_loss = np.mean(batch_val_losses)
        val_loss_history.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            if save_final_model: torch.save(model.state_dict(), "best_model_checkpoint.pth")
        else:
            epochs_no_improve += 1
            
        if (epoch + 1) % 10 == 0: print(f"Epoch [{epoch+1}/{EPOCHS}], Val Loss: {epoch_val_loss:.6f}")
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    if save_final_model: 
        plot_training_history(train_loss_history, val_loss_history)
        model.load_state_dict(torch.load("best_model_checkpoint.pth"))

    model.eval()
    val_errors = []
    with torch.no_grad():
        for seq_val, _ in val_loader:
            reconstructed = model(seq_val.to(device))
            error = torch.mean((seq_val.to(device) - reconstructed) ** 2, dim=[1,2])
            val_errors.extend(error.cpu().numpy())
    if not val_errors: return 0.0
    reconstruction_threshold = np.percentile(val_errors, THRESHOLD_PERCENTILE)
    print(f"\nReconstruction Threshold ({THRESHOLD_PERCENTILE}th percentile): {reconstruction_threshold:.6f}")
    
    test_sequences_np = prepared_data['test_sequences']
    true_labels = prepared_data['test_labels']
    
    test_loader = DataLoader(SequenceDataset(test_sequences_np), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    all_predictions = []
    with torch.no_grad():
        for seq_batch, _ in test_loader:
            reconstructed = model(seq_batch.to(device))
            error = torch.mean((seq_batch.to(device) - reconstructed) ** 2, dim=[1, 2])
            all_predictions.extend((error > reconstruction_threshold).cpu().numpy().astype(int))

    if not all_true_labels: return 0.0
    f1 = f1_score(all_true_labels, all_predictions, pos_label=1, zero_division=0)
    if trial_number is not None: print(f"Trial #{trial_number} Overall Anomaly F1-Score: {f1:.4f}")
    
    if save_final_model:
        print("\n--- Final Performance Report ---")
        print(classification_report(all_true_labels, all_predictions, target_names=['Normal', 'Anomaly'], zero_division=0))
        cm = confusion_matrix(all_true_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Normal', 'Predicted Anomaly'], yticklabels=['Actual Normal', 'Actual Anomaly'])
        plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title('Final Model Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        os.rename("best_model_checkpoint.pth", "anomaly_autoencoder.pth")
        joblib.dump(prepared_data['scaler'], "anomaly_scaler.pkl")
        params['reconstruction_threshold'] = reconstruction_threshold
        with open("best_params.json", 'w') as f:
            json.dump(params, f, indent=4)
        print("Artifacts saved.")
    return f1