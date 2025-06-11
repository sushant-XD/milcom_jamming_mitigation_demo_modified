# tune_hyperparams.py (Corrected import)
import optuna
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import joblib
# Import both functions from model.py
from model import run_training_and_evaluation, create_sequences

# --- Data Preparation Functions (Centralized Here) ---
def load_all_data(root_dir, prefix):
    all_dfs = []
    print(f"-> Loading files with prefix: '{prefix}'...")
    for root, _, files in os.walk(DATA_DIR):
        for filename in files:
            if filename.startswith(prefix) and filename.endswith('.csv'):
                file_path = os.path.join(root, filename)
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    if not df.empty:
                        all_dfs.append(df)
                except Exception as e:
                    print(f"  -> Could not read {filename}: {e}")
    if not all_dfs: return None
    return pd.concat(all_dfs, ignore_index=True)

def process_and_clean(df, features):
    if df is None or df.empty: return None
    # Filter for ue2 first, as it's the primary device for training
    df_ue2 = df[df['ue_identifier'] == 'ue2'].copy()
    for col in features:
        df_ue2[col] = pd.to_numeric(df_ue2[col], errors='coerce')
    df_ue2.dropna(subset=features, inplace=True)
    return df_ue2

# --- Objective Function for Optuna ---
def objective(trial, prepared_data):
    params = {
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'epochs': 300, 
        'early_stopping_patience': 15,
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
        'latent_size': trial.suggest_int('latent_size', 8, 64),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'threshold_percentile': trial.suggest_int('threshold_percentile', 90, 99),
        'sequence_length': prepared_data['sequence_length']
    }
    f1_score = run_training_and_evaluation(params, prepared_data, save_final_model=False, trial_number=trial.number)
    return f1_score

# --- Main Execution Block ---
if __name__ == "__main__":
    DATA_DIR = "srsran_csv_output"
    FEATURES = ['dl_cqi', 'ul_phr', 'dl_percent', 'ul_percent', 'dl_nok']
    SEQUENCE_LENGTH = 15 

    # 1. Load and Clean Data (ONCE)
    print("--- Starting One-Time Data Preparation ---")
    normal_df_raw = load_all_data(DATA_DIR, 'no_jammer')
    jammed_df_raw = load_all_data(DATA_DIR, 'jammer')
    
    if normal_df_raw is None or jammed_df_raw is None:
        print("Critical Error: Could not load data. Exiting.")
        exit()

    normal_ue2_df = process_and_clean(normal_df_raw, FEATURES)
    jammed_ue2_df = process_and_clean(jammed_df_raw, FEATURES)
    
    if normal_ue2_df.empty or jammed_ue2_df.empty:
        print("Critical Error: Data for normal or jammed UE2 is empty after cleaning. Exiting.")
        exit()

    # 2. Scale and Split Training Data
    scaler = MinMaxScaler()
    X_normal_ue2_scaled = scaler.fit_transform(normal_ue2_df[FEATURES].astype(float))
    
    train_data_scaled, val_data_scaled = train_test_split(X_normal_ue2_scaled, test_size=0.2, random_state=42)
    
    # 3. Create All Necessary Sequences (ONCE)
    prepared_data = {
        'train': create_sequences(train_data_scaled, SEQUENCE_LENGTH),
        'validation': create_sequences(val_data_scaled, SEQUENCE_LENGTH),
        'scaler': scaler,
        'sequence_length': SEQUENCE_LENGTH
    }
    
    # Create test sequences for all data categories
    X_jammed_ue2_scaled = scaler.transform(jammed_ue2_df[FEATURES].astype(float))
    
    normal_sequences = create_sequences(X_normal_ue2_scaled, SEQUENCE_LENGTH)
    jammed_sequences = create_sequences(X_jammed_ue2_scaled, SEQUENCE_LENGTH)
    
    prepared_data['test_sequences'] = np.concatenate([normal_sequences, jammed_sequences])
    prepared_data['test_labels'] = np.concatenate([np.zeros(len(normal_sequences)), np.ones(len(jammed_sequences))])
    
    print(f"\nData preparation complete. {len(prepared_data['train'])} training sequences created.")
    
    # 4. Start Tuning
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    print("\n--- Starting Hyperparameter Tuning ---")
    study.optimize(lambda trial: objective(trial, prepared_data), n_trials=100)
    
    # 5. Report and Save Best Results
    print("\n--- Hyperparameter Tuning Results ---")
    print(f"Number of finished trials: {len(study.trials)}")
    best_trial = study.best_trial
    print(f"Best trial achieved F1-score: {best_trial.value:.4f}")
    print("Best parameters found:")
    best_params = best_trial.params
    for key, value in best_params.items():
        print(f"  {key}: {value}")
        
    with open("best_params_from_tuning.json", 'w') as f:
        json.dump(best_params, f, indent=4)
        
    print("\nBest parameters saved to 'best_params_from_tuning.json'")