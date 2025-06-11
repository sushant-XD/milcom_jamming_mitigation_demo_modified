import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration (remains the same) ---
DATA_DIR = "srsran_csv_output"
UE_IDENTIFIER = "ue2"
DISTANCE_THRESHOLD_FT = 25
SEQUENCE_LENGTH = 10
FEATURES = ['dl_cqi', 'cqi_diff', 'cqi_rolling_std']

# --- Helper Functions (load_all_data and prepare_supervised_data are the same) ---
def load_all_data(root_dir, prefix, ue_filter):
    # (This function is the same as before)
    all_dfs = []
    # print(f"Searching for all '{prefix}*.csv' files for '{ue_filter}' in '{root_dir}'...")
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
    if not all_dfs: return None
    return pd.concat(all_dfs, ignore_index=True)

def prepare_supervised_data(df, label):
    # (This function is the same as before, but we add 'ue2_distance_ft' to the output)
    
    # Feature Engineering
    df['dl_cqi'] = pd.to_numeric(df['dl_cqi'], errors='coerce')
    df['ue2_distance_ft'] = pd.to_numeric(df['ue2_distance_ft'], errors='coerce')
    df.sort_index(inplace=True)
    df['cqi_diff'] = df['dl_cqi'].diff().fillna(0)
    df['cqi_rolling_std'] = df['dl_cqi'].rolling(window=5).std().fillna(0)
    df.dropna(subset=FEATURES + ['ue2_distance_ft'], inplace=True)

    # We need to create sequences from the distance column as well to keep alignment
    feature_data = df[FEATURES].values
    distance_data = df['ue2_distance_ft'].values
    
    # Create sequences
    sequences, sequence_distances = [], []
    if len(feature_data) <= SEQUENCE_LENGTH:
        return np.array([]), np.array([]), np.array([])
        
    for i in range(len(feature_data) - SEQUENCE_LENGTH):
        sequences.append(feature_data[i:(i + SEQUENCE_LENGTH)])
        # The distance for a sequence can be the last known distance in that window
        sequence_distances.append(distance_data[i + SEQUENCE_LENGTH - 1])
    
    X = np.array(sequences)
    y = np.full(len(X), label)
    dist = np.array(sequence_distances)
    
    return X, y, dist # Return distances as well


if __name__ == "__main__":
    # 1. Load ALL data
    normal_df_all = load_all_data(DATA_DIR, 'no_jammer', UE_IDENTIFIER)
    jammed_df_all = load_all_data(DATA_DIR, 'jammer', UE_IDENTIFIER)

    if normal_df_all is None or jammed_df_all is None:
        print("\nERROR: Could not load both normal and jammed datasets. Exiting.")
        exit()

    # 2. Filter for the difficult, low-SNR environment
    print(f"\n--- Filtering data for distances > {DISTANCE_THRESHOLD_FT} ft ---")
    normal_low_snr = normal_df_all[pd.to_numeric(normal_df_all['ue2_distance_ft'], errors='coerce') > DISTANCE_THRESHOLD_FT]
    jammed_low_snr = jammed_df_all[pd.to_numeric(jammed_df_all['ue2_distance_ft'], errors='coerce') > DISTANCE_THRESHOLD_FT]

    if normal_low_snr.empty or jammed_low_snr.empty:
        print("Not enough data beyond the distance threshold to run the supervised model. Exiting.")
        exit()
        
    print(f"Found {len(normal_low_snr)} normal rows and {len(jammed_low_snr)} jammed rows.")

    # 3. Prepare the datasets
    # ### MODIFICATION: Capture the distance for each sequence ###
    X_normal, y_normal, dist_normal = prepare_supervised_data(normal_low_snr, label=0)
    X_jammed, y_jammed, dist_jammed = prepare_supervised_data(jammed_low_snr, label=1)

    if len(X_normal) == 0 or len(X_jammed) == 0:
        print("Not enough data to create sequences after filtering. Exiting.")
        exit()

    # 4. Combine and Split Data
    X_combined = np.concatenate((X_normal, X_jammed), axis=0)
    y_combined = np.concatenate((y_normal, y_jammed), axis=0)
    dist_combined = np.concatenate((dist_normal, dist_jammed), axis=0) # Also combine distances
    
    n_samples, seq_len, n_features = X_combined.shape
    X_flattened = X_combined.reshape((n_samples, seq_len * n_features))

    print(f"\nTotal samples: {n_samples}, features per sample: {seq_len * n_features}")

    # Use train_test_split on all arrays to keep them aligned
    X_train, X_test, y_train, y_test, dist_train, dist_test = train_test_split(
        X_flattened, y_combined, dist_combined, test_size=0.3, random_state=42, stratify=y_combined
    )

    # 5. Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Train the Supervised Model
    print("\n--- Training RandomForestClassifier ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # 7. Evaluate the Model (with distance breakdown)
    print("\n" + "="*80)
    print("### OVERALL PERFORMANCE (ACROSS ALL TEST DISTANCES) ###")
    print("="*80)
    
    y_pred_overall = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_overall, target_names=['Normal (Class 0)', 'Jammed (Class 1)']))

    # ### MODIFICATION: Loop through each unique distance in the test set ###
    print("\n" + "="*80)
    print("### PERFORMANCE BREAKDOWN BY DISTANCE ###")
    print("="*80)
    
    unique_test_distances = sorted(np.unique(dist_test))
    
    for distance in unique_test_distances:
        print(f"\n--- Results for Distance: {int(distance)} ft ---")
        
        # Create a boolean mask to select only the data for the current distance
        distance_mask = (dist_test == distance)
        
        # Select the subset of test data and labels
        X_test_subset = X_test_scaled[distance_mask]
        y_test_subset = y_test[distance_mask]
        
        if len(y_test_subset) == 0:
            print("No test samples for this distance.")
            continue
            
        # Make predictions on this subset
        y_pred_subset = model.predict(X_test_subset)
        
        # Print the classification report for this subset
        print(classification_report(y_test_subset, y_pred_subset, target_names=['Normal', 'Jammed'], zero_division=0))

    # (Saving logic remains the same)
    joblib.dump(model, 'supervised_random_forest.pkl')
    joblib.dump(scaler, 'supervised_scaler.pkl')
    print("\nSaved supervised model and scaler.")