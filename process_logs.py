import os
import re
import csv

def parse_srsran_log(file_path):
    """
    Parses a srsRAN log file by generating data for the idle 'ue1'.
    - Finds any row with traffic and labels it 'ue2'.
    - For each 'ue2' row, it manufactures a corresponding 'ue1' row with zeroed-out metrics.
    This ensures both UEs are represented in the final dataset at every time step.
    """
    metadata = {
        'ue1_distance_ft': None, 'ue2_distance_ft': None, 'gain_db': None,
        'amp': None, 'amp_bandwidth': None, 'jammer_gnb_distance_ft': None,
    }
    data_rows = []

    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  -> Could not read file '{os.path.basename(file_path)}'. Error: {e}")
        return None, [], []

    # --- Regex for METADATA ONLY ---
    ue_re = re.compile(r'(ue\d+)\s*=\s*(\d+)')
    distance_re = re.compile(r'\(\s*(\d+)\s*ft\s*\)')
    gain_re = re.compile(r'gain\s+([\d\.]+)\s*db', re.IGNORECASE)
    amp_re = re.compile(r'amp\s+([\d\.]+)', re.IGNORECASE)
    amp_bandwidth_re = re.compile(r'amp_bandwidth:\s*([\d\.]+)', re.IGNORECASE)
    jammer_gnb_re = re.compile(r'jammer gnb\s*=\s*(\d+)ft', re.IGNORECASE)

    table_headers = [
        'pci', 'ue_identifier', 'dl_cqi', 'dl_ri', 'dl_mcs', 'dl_brate', 'dl_ok', 'dl_nok', 'dl_percent', 'dl_bs',
        'ul_pusch', 'ul_rsrp', 'ul_ri', 'ul_mcs', 'ul_brate', 'ul_ok', 'ul_nok', 'ul_percent', 'ul_bsr', 'ul_ta', 'ul_phr'
    ]
    is_data_section = False

    for line in lines:
        if "==== gNB started ===" in line:
            is_data_section = True
            continue

        if not is_data_section:
            # Parse header for metadata (e.g., distances)
            m = ue_re.search(line)
            if m:
                ue_id, rnti = m.groups()
                dist_m = distance_re.search(line)
                if dist_m and ue_id in ['ue1', 'ue2']:
                    metadata[f"{ue_id}_distance_ft"] = dist_m.group(1)
            if gain_re.search(line): metadata['gain_db'] = gain_re.search(line).group(1)
            if amp_re.search(line): metadata['amp'] = amp_re.search(line).group(1)
            if amp_bandwidth_re.search(line): metadata['amp_bandwidth'] = amp_bandwidth_re.search(line).group(1)
            if jammer_gnb_re.search(line): metadata['jammer_gnb_distance_ft'] = jammer_gnb_re.search(line).group(1)

        else: # This is the data section
            if re.match(r'\s+\d', line):
                parts = re.split(r'\s+', line.replace('|', '').strip())

                if len(parts) >= 21: # Ensure we have all columns
                    dl_brate = parts[5]

                    # If this row has traffic, it MUST be ue2.
                    if not dl_brate.startswith('0.0'):
                        # --- PROCESS THE UE2 (ACTIVE) ROW ---
                        ue2_row = list(parts)
                        ue2_row[1] = 'ue2' # Set identifier

                        # Clean the data
                        for i in [5, 14]:
                            val = ue2_row[i].lower()
                            if 'k' in val: ue2_row[i] = str(float(val.replace('k', '')) * 1000)
                            elif 'm' in val: ue2_row[i] = str(float(val.replace('m', '')) * 1000000)
                        if 'n' in ue2_row[19].lower(): ue2_row[19] = ue2_row[19].lower().replace('n', '')
                        for i in [8, 17]: ue2_row[i] = ue2_row[i].replace('%', '')
                        
                        data_rows.append(ue2_row)

                        # --- MANUFACTURE THE UE1 (IDLE) ROW ---
                        # Create a copy to use as a template
                        ue1_row = list(ue2_row)
                        ue1_row[1] = 'ue1' # Set identifier

                        # Zero out all performance metrics from dl_cqi to ul_phr
                        # These are indices 2 through 20
                        for i in range(2, 21):
                            ue1_row[i] = '0'

                        data_rows.append(ue1_row)

    return metadata, data_rows, table_headers


def process_directory(input_dir, output_dir):
    # This function is correct and does not need changes.
    metadata_headers = ['ue1_distance_ft', 'ue2_distance_ft', 'gain_db', 'amp', 'amp_bandwidth', 'jammer_gnb_distance_ft']
    
    for root, dirs, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        current_output_dir = os.path.join(output_dir, relative_path) if relative_path != '.' else output_dir
        if not os.path.exists(current_output_dir): os.makedirs(current_output_dir)

        for filename in files:
            if filename.startswith('.'): continue
            
            file_path = os.path.join(root, filename)
            csv_path = os.path.join(current_output_dir, os.path.splitext(filename)[0] + ".csv")

            print(f"Processing '{file_path}'...")
            metadata, data_rows, table_headers = parse_srsran_log(file_path)

            if not data_rows:
                print(f"  -> No data rows with traffic found in '{filename}'. Skipping.")
                continue

            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(metadata_headers + table_headers)
                metadata_values = [metadata.get(h) or '' for h in metadata_headers]
                for row in data_rows:
                    writer.writerow(metadata_values + row)
            print(f"  -> Successfully created '{csv_path}'")

# --- Main Execution ---
if __name__ == "__main__":
    input_directory = "dataset/NoJammer"
    output_directory = "srsran_csv_output"
    if not os.path.exists(input_directory):
        print(f"Input directory '{input_directory}' not found.")
    else:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        process_directory(input_directory, output_directory)
        print("\nProcessing complete.")
