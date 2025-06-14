import os
import json
import optuna
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report

def load_and_process_csvs_from_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: Directory not found at '{folder_path}'")
        return None
    all_dfs = []
    print(f"Searching for all '.csv' files in '{folder_path}'...")
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.csv'):
                file_path = os.path.join(root, filename)
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    if not df.empty:
                        all_dfs.append(df)
                except Exception as e:
                    print(f"  -> Could not read or process {filename}. Error: {e}")
    if not all_dfs:
        print(f"  -> No non-empty CSV files found in '{folder_path}'.")
        return None
    return pd.concat(all_dfs, ignore_index=True)

class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.sequences[idx]

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, sequence_len, hidden_size, latent_size, dropout_rate):
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
        out_lstm1, _ = self.encoder_lstm1(x); out_dropout1 = self.encoder_dropout1(out_lstm1); out_relu1 = self.encoder_relu1(out_dropout1); out_lstm2, (h_n_encoder, c_n_encoder) = self.encoder_lstm2(out_relu1); out_dropout2 = self.encoder_dropout2(out_lstm2); encoded_sequence_activated = self.encoder_relu2(out_dropout2); latent_vector = encoded_sequence_activated[:, -1, :]; decoder_input_sequence = latent_vector.unsqueeze(1).repeat(1, x.size(1), 1); out_lstm3, _ = self.decoder_lstm1(decoder_input_sequence); out_dropout3 = self.decoder_dropout3(out_lstm3); out_relu3 = self.decoder_relu1(out_dropout3); reconstructed_seq, _ = self.decoder_lstm2(out_relu3); return reconstructed_seq

# --- The Objective Function for Optuna ---
def objective(trial, prepared_data):
    """
    This function takes a trial from Optuna, trains a model with the suggested
    hyperparameters, and returns the performance metric we want to maximize.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Define the search space for hyperparameters
    params = {
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128]),
        'latent_size': trial.suggest_int('latent_size', 16, 64),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'threshold_percentile': trial.suggest_int('threshold_percentile', 85, 99)
    }

    # 2. Prepare DataLoaders for this trial
    train_loader = DataLoader(SequenceDataset(prepared_data['train_sequences']), batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(SequenceDataset(prepared_data['val_sequences']), batch_size=params['batch_size'], shuffle=False)
    
    # 3. Instantiate and Train the Model
    model = LSTMAutoencoder(
        input_size=prepared_data['input_size'],
        sequence_len=prepared_data['sequence_length'],
        hidden_size=params['hidden_size'],
        latent_size=params['latent_size'],
        dropout_rate=params['dropout_rate']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    best_val_loss, epochs_no_improve = float('inf'), 0
    for epoch in range(500): # Use a fixed, reasonable number of epochs for tuning
        model.train()
        for seq, _ in train_loader:
            seq_on_device = seq.to(device)
            reconstructed = model(seq_on_device)
            loss = criterion(reconstructed, seq_on_device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            batch_val_losses = [criterion(model(seq.to(device)), seq.to(device)).item() for seq, _ in val_loader]
            epoch_val_loss = np.mean(batch_val_losses)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss, epochs_no_improve = epoch_val_loss, 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= 10: # Shorter patience for faster tuning
            break
            
    # 4. Evaluate the trained model to get the F1-score
    model.eval()
    val_errors = []
    with torch.no_grad():
        for seq, _ in val_loader:
            reconstructed = model(seq.to(device))
            loss_per_item = torch.mean((reconstructed - seq.to(device)) ** 2, dim=[1, 2])
            val_errors.extend(loss_per_item.cpu().numpy())
    val_errors = np.array(val_errors)
    
    global_threshold = np.percentile(val_errors, params['threshold_percentile'])
    thresholds = {}
    for distance in prepared_data['common_distances']:
        distance_specific_errors = val_errors[prepared_data['val_distances'] == distance]
        if len(distance_specific_errors) > 0:
            thresholds[distance] = np.percentile(distance_specific_errors, params['threshold_percentile'])
        else:
            thresholds[distance] = global_threshold

    test_loader = DataLoader(SequenceDataset(prepared_data['test_sequences']), batch_size=params['batch_size'], shuffle=False)
    test_errors = []
    with torch.no_grad():
        for seq, _ in test_loader:
            reconstructed = model(seq.to(device))
            loss_per_item = torch.mean((reconstructed - seq.to(device)) ** 2, dim=[1, 2])
            test_errors.extend(loss_per_item.cpu().numpy())
    test_errors = np.array(test_errors)
    
    predictions = np.array([1 if test_errors[i] > thresholds.get(prepared_data['all_distances_per_seq'][i], global_threshold) else 0 for i in range(len(test_errors))])
    
    # 5. Calculate and Return the Target Metric
    report = classification_report(
        prepared_data['true_labels'],
        predictions,
        output_dict=True,
        zero_division=0
    )
    
    # We will target the F1-score for anomalies specifically at 30ft, as it's our most reliable indicator
    mask_30ft = prepared_data['all_distances_per_seq'] == 30
    if np.sum(mask_30ft) > 0:
        report_30ft = classification_report(
            prepared_data['true_labels'][mask_30ft],
            predictions[mask_30ft],
            output_dict=True,
            zero_division=0
        )
        f1_score_30ft = report_30ft.get('Anomaly', {}).get('f1-score', 0.0)
        return f1_score_30ft
    else:
        # If no 30ft data, return 0
        return 0.0

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- 1. One-Time Data Preparation ---
    print("--- Starting One-Time Data Preparation ---")
    normal_df = load_and_process_csvs_from_folder("no_jamming_processed_data")
    jammed_df = load_and_process_csvs_from_folder("jamming_processed_data")
    if normal_df is None or jammed_df is None: exit()

    FEATURES = ['dl_cqi', 'ul_rsrp', 'cqi_diff', 'cqi_rolling_std', 'quality_divergence', 'distance_ft']
    FEATURES_TO_SCALE = ['dl_cqi', 'ul_rsrp']

    temp_normal_rsrp = pd.to_numeric(normal_df['ul_rsrp'], errors='coerce'); temp_jammed_rsrp = pd.to_numeric(jammed_df['ul_rsrp'], errors='coerce'); max_rsrp = pd.concat([temp_normal_rsrp, temp_jammed_rsrp]).max(); overload_value = max_rsrp + 10
    print(f"Max measured RSRP: {max_rsrp:.2f}. Using {overload_value:.2f} for 'ovl'.")

    temp_normal_clean = normal_df.copy(); temp_normal_clean['ul_rsrp'].replace('ovl', overload_value, inplace=True); temp_normal_clean['ul_rsrp'] = pd.to_numeric(temp_normal_clean['ul_rsrp'], errors='coerce'); temp_normal_clean['dl_cqi'] = pd.to_numeric(temp_normal_clean['dl_cqi'], errors='coerce'); temp_normal_clean.dropna(subset=['ul_rsrp', 'dl_cqi'], inplace=True)
    scaler = RobustScaler().fit(temp_normal_clean[FEATURES_TO_SCALE])
    
    all_dfs = {'Normal': normal_df, 'Jammed': jammed_df}
    common_distances = None
    for name, df in all_dfs.items():
        df['ul_rsrp'].replace('ovl', overload_value, inplace=True); df['ul_rsrp'] = pd.to_numeric(df['ul_rsrp'], errors='coerce'); df['ul_rsrp'].ffill(inplace=True); df['ul_rsrp'].bfill(inplace=True); df['distance_ft'] = pd.to_numeric(df['distance_ft'], errors='coerce'); df['dl_cqi'] = pd.to_numeric(df['dl_cqi'], errors='coerce');
        df.dropna(subset=['distance_ft', 'dl_cqi', 'ul_rsrp'], inplace=True); df['distance_ft'] = df['distance_ft'].astype(int); df[FEATURES_TO_SCALE] = scaler.transform(df[FEATURES_TO_SCALE]); df['quality_divergence'] = df['ul_rsrp'] - df['dl_cqi'];
        df.sort_values(by='distance_ft', kind='mergesort', inplace=True); df['cqi_diff'] = df.groupby('distance_ft')['dl_cqi'].diff().fillna(0); df['cqi_rolling_std'] = df.groupby('distance_ft')['dl_cqi'].rolling(window=5).std().fillna(0).reset_index(level=0, drop=True); df.dropna(subset=FEATURES, inplace=True)
        distances = set(df['distance_ft'].unique());
        if common_distances is None: common_distances = distances
        else: common_distances.intersection_update(distances)
    common_distances = sorted(list(common_distances))

    # --- 2. Create All Necessary Sequences (ONCE) ---
    def create_sequences_with_distance(df, features, sequence_length, distance_col='distance_ft'):
        data_values = df[features].values
        sequences = [];
        if len(data_values) > sequence_length:
            for i in range(len(data_values) - sequence_length): sequences.append(data_values[i:(i + sequence_length)])
        sequences = np.array(sequences)
        if len(sequences) == 0: return sequences, np.array([])
        distances = [df[distance_col].iloc[i + sequence_length - 1] for i in range(len(data_values) - sequence_length)]
        return sequences, np.array(distances)
        
    normal_sequences, normal_distances = create_sequences_with_distance(normal_df, FEATURES, 10)
    jammed_sequences, jammed_distances = create_sequences_with_distance(jammed_df, FEATURES, 10)
    
    train_sequences, val_sequences, train_distances, val_distances = train_test_split(
        normal_sequences, normal_distances, test_size=0.2, random_state=42
    )

    # Package data for the objective function
    prepared_data = {
        'train_sequences': train_sequences,
        'val_sequences': val_sequences,
        'val_distances': val_distances,
        'test_sequences': np.concatenate([normal_sequences, jammed_sequences]),
        'true_labels': np.concatenate([np.zeros(len(normal_sequences)), np.ones(len(jammed_sequences))]),
        'all_distances_per_seq': np.concatenate([normal_distances, jammed_distances]),
        'common_distances': common_distances,
        'sequence_length': 10,
        'input_size': len(FEATURES),
    }
    
    print(f"\nData preparation complete. {len(prepared_data['train_sequences'])} training sequences created.")

    # --- 3. Start Tuning ---
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    print("\n--- Starting Hyperparameter Tuning ---")
    study.optimize(lambda trial: objective(trial, prepared_data), n_trials=50) # Run 50 trials
    
    # --- 4. Report and Save Best Results ---
    print("\n--- Hyperparameter Tuning Results ---")
    print(f"Number of finished trials: {len(study.trials)}")
    best_trial = study.best_trial
    print(f"Best trial achieved F1-score (at 30ft): {best_trial.value:.4f}")
    print("Best parameters found:")
    best_params = best_trial.params
    for key, value in best_params.items():
        print(f"  {key}: {value}")
        
    # Add the fixed sequence length to the params file for completeness
    best_params['sequence_length'] = 10
    with open("best_params.json", 'w') as f:
        json.dump(best_params, f, indent=4)
        
    print("\nBest parameters saved to 'best_params.json'")
