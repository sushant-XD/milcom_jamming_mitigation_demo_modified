# preprocess_and_cache.py (Corrected)
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# --- Configuration ---
DATA_DIR = "srsran_csv_output"
CACHE_DIR = "data_cache"
FEATURES = ['dl_cqi', 'ul_phr', 'dl_percent', 'ul_percent', 'dl_nok']
# NOTE: If you tune sequence_length later, you must re-run this script with the new value.
SEQUENCE_LENGTH = 15 

# --- Helper Functions ---
def load_all_data(prefix):
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
    
    if not all_dfs:
        print(f"  -> No data found for prefix '{prefix}'.")
        return None
        
    return pd.concat(all_dfs, ignore_index=True)

def process_and_clean(df):
    if df is None or df.empty:
        return None
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=FEATURES, inplace=True)
    return df

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:(i + sequence_length)])
    return np.array(sequences)

# --- Main Pre-processing Logic ---
if __name__ == "__main__":
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print(f"Created cache directory: {CACHE_DIR}")

    # 1. Load all data
    normal_df_raw = load_all_data('no_jammer')
    
    # THIS IS THE CORRECTED LINE
    jammed_df_raw = load_all_data('jammer')

    # 2. Clean and separate UE data
    normal_ue1_df = process_and_clean(normal_df_raw[normal_df_raw['ue_identifier'] == 'ue1'].copy())
    normal_ue2_df = process_and_clean(normal_df_raw[normal_df_raw['ue_identifier'] == 'ue2'].copy())
    jammed_ue1_df = process_and_clean(jammed_df_raw[jammed_df_raw['ue_identifier'] == 'ue1'].copy())
    jammed_ue2_df = process_and_clean(jammed_df_raw[jammed_df_raw['ue_identifier'] == 'ue2'].copy())

    # 3. Prepare training and validation data (ONLY from normal_ue2)
    print("\n--- Preparing Training and Validation Sets (from normal_ue2) ---")
    if normal_ue2_df is not None and not normal_ue2_df.empty:
        X_train_full = normal_ue2_df[FEATURES].astype(float)
        scaler = MinMaxScaler()
        X_train_full_scaled = scaler.fit_transform(X_train_full)
        
        train_data_scaled, val_data_scaled = train_test_split(X_train_full_scaled, test_size=0.2, random_state=42)

        # Create and save sequences
        train_sequences = create_sequences(train_data_scaled, SEQUENCE_LENGTH)
        val_sequences = create_sequences(val_data_scaled, SEQUENCE_LENGTH)
        
        print(f"Created {len(train_sequences)} training sequences.")
        print(f"Created {len(val_sequences)} validation sequences.")

        joblib.dump(train_sequences, os.path.join(CACHE_DIR, 'train_sequences.pkl'))
        joblib.dump(val_sequences, os.path.join(CACHE_DIR, 'val_sequences.pkl'))
        joblib.dump(scaler, os.path.join(CACHE_DIR, 'scaler.pkl'))
        print("Saved train/val sequences and scaler to cache.")
    else:
        print("ERROR: No normal_ue2 data found. Cannot create training cache.")

    # 4. Prepare full test sets
    print("\n--- Preparing Full Test Sets ---")
    
    datasets_to_cache = {
        "normal_ue1_test": normal_ue1_df,
        "jammed_ue1_test": jammed_ue1_df,
        "jammed_ue2_test": jammed_ue2_df,
    }

    # Also use all of normal_ue2_df for final testing
    if normal_ue2_df is not None and not normal_ue2_df.empty:
        # We already scaled this data when fitting the scaler
        # We just need to create sequences from the whole set
        normal_ue2_sequences = create_sequences(X_train_full_scaled, SEQUENCE_LENGTH)
        joblib.dump(normal_ue2_sequences, os.path.join(CACHE_DIR, 'normal_ue2_test_sequences.pkl'))
        print(f"Saved {len(normal_ue2_sequences)} normal_ue2_test sequences.")

    for name, df in datasets_to_cache.items():
        if df is not None and not df.empty:
            X_test_scaled = scaler.transform(df[FEATURES].astype(float))
            sequences = create_sequences(X_test_scaled, SEQUENCE_LENGTH)
            joblib.dump(sequences, os.path.join(CACHE_DIR, f'{name}_sequences.pkl'))
            print(f"Saved {len(sequences)} sequences for {name}.")

    print("\nPreprocessing and caching complete.")