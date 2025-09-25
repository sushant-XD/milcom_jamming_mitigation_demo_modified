import csv
import re
import argparse
import sys

def clean_brate(value):
    """Converts bitrate strings (e.g., '1.7k', '7.2M') to a numeric string."""
    value = value.lower().strip()
    if not value or value == 'n/a':
        return value
    
    try:
        if value.endswith('k'):
            return str(int(float(value[:-1]) * 1000))
        if value.endswith('m'):
            return str(int(float(value[:-1]) * 1000000))
        if value.endswith('g'):
            return str(int(float(value[:-1]) * 1000000000))
        
        return str(int(float(value))) 
    except (ValueError, TypeError):
        return value

def process_log_file(input_path, output_path):
    print(f"--- Starting processing for {input_path} ---")
    
    # MODIFIED: Added a map to link RNTI to UE name
    ue_to_distance_map = {}
    rnti_to_ue_map = {}
    
    # MODIFIED: New pattern to capture UE name, RNTI, and distance
    mapping_pattern = re.compile(r'(ue\d+)\s*=\s*(\S+).*?\(([\d.]+)ft\)')
    
    processing_state = 'HEADER'

    try:
        with open(input_path, 'r') as infile, open(output_path, 'w', newline='') as outfile:
            csv_writer = csv.writer(outfile)
    
            csv_header = [
                'pci', 'distance_ft', 'dl_cqi', 'dl_ri', 'dl_mcs', 'dl_brate', 'dl_ok', 'dl_nok', 'dl_percent', 'dl_bs',
                'ul_pusch', 'ul_rsrp', 'ul_ri', 'ul_mcs', 'ul_brate', 'ul_ok', 'ul_nok', 'ul_percent', 'ul_bsr', 'ul_ta', 'ul_phr'
            ]
            csv_writer.writerow(csv_header)
    
            for line in infile:
                if processing_state == 'HEADER':
                    match = mapping_pattern.search(line)
                    if match:
                        ue_name, rnti, distance = match.groups()
                        
                        # Clean up the RNTI if it has extra characters
                        rnti = rnti.strip('()')

                        if distance.endswith('.0'):
                            distance = distance[:-2]
                        
                        # MODIFIED: Populate both maps
                        ue_to_distance_map[ue_name] = distance
                        rnti_to_ue_map[rnti] = ue_name
                        continue

                    if 'pci rnti' in line and '|' in line:
                        print(f"Found UE-to-distance mapping: {ue_to_distance_map}")
                        print(f"Found RNTI-to-UE mapping: {rnti_to_ue_map}") # For debugging
                        if not ue_to_distance_map:
                            print("FATAL: Could not find any UE mappings in the header. Exiting.", file=sys.stderr)
                            sys.exit(1)
                        print("Switching to DATA processing state.")
                        processing_state = 'DATA'
                        continue 

                if processing_state == 'DATA':
                    line = line.strip()

                    if '|' not in line or '---' in line or 'pci rnti' in line:
                        continue
                    
                    try:
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) != 3:
                            continue

                        pci_rnti_part = parts[0].split()
                        dl_part = parts[1].split()
                        ul_part = parts[2].split()
                        
                        all_values = pci_rnti_part + dl_part + ul_part
                        
                        if len(all_values) != 21:
                            continue

                        # MODIFIED: Perform a two-step lookup
                        rnti = all_values[1] 
                        ue_name = rnti_to_ue_map.get(rnti)
                        
                        if ue_name is None:
                            # This RNTI wasn't defined in the header, so we skip it
                            continue
                        
                        distance = ue_to_distance_map.get(ue_name)
                        if distance is None:
                            continue
                        
                        row_data = [
                            all_values[0],  # pci
                            distance,       # distance_ft (Now correctly looked up)
                            all_values[2],  # dl_cqi
                            all_values[3],  # dl_ri
                            all_values[4],  # dl_mcs
                            clean_brate(all_values[5]),  # dl_brate
                            all_values[6],  # dl_ok
                            all_values[7],  # dl_nok
                            all_values[8].rstrip('%'),   # dl_percent
                            all_values[9],  # dl_bs
                            all_values[10], # ul_pusch
                            all_values[11], # ul_rsrp
                            all_values[12], # ul_ri
                            all_values[13], # ul_mcs
                            clean_brate(all_values[14]), # ul_brate
                            all_values[15], # ul_ok
                            all_values[16], # ul_nok
                            all_values[17].rstrip('%'),  # ul_percent
                            clean_brate(all_values[18]), # ul_bsr
                            all_values[19].rstrip('n'),  # ul_ta
                            all_values[20], # ul_phr
                        ]
                        csv_writer.writerow(row_data)

                    except (IndexError, ValueError):
                        continue
        
        print(f"\n--- Successfully created CSV file: {output_path} ---")

    except FileNotFoundError:
        print(f"Error: The input file '{input_path}' was not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parse srsRAN gNB log files into a clean CSV format. This script processes the file in a single pass.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_file", help="Path to the input srsRAN log file.")
    parser.add_argument("output_file", help="Path for the output CSV file.")
    
    args = parser.parse_args()
    
    process_log_file(args.input_file, args.output_file)
