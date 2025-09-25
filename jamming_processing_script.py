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
    """
    Parses the srsRAN log file in a single pass, extracting UE distances and jammer distance.
    """
    print(f"--- Starting processing for {input_path} ---")
    
    # MODIFIED: Added a map to link RNTI to UE name
    ue_to_distance_map = {}
    rnti_to_ue_map = {}
    jammer_distance = 'n/a'  

    processing_state = 'HEADER'

    # MODIFIED: New pattern to capture UE name, RNTI, and distance
    ue_mapping_pattern = re.compile(r'(ue\d+)\s*=\s*(\S+).*?\(([\d.]+)ft\)')
    
    jammer_dist_pattern = re.compile(r'jammer.*\s(\d+)ft', re.IGNORECASE)

    data_line_pattern = re.compile(
        r"^\s*(\S+)\s+(\S+)\s*\|\s*"  # pci, rnti
        r"(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*\|\s*"  # DL block (8 fields)
        r"(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)"  # UL block (11 fields)
    )

    try:
        with open(input_path, 'r') as infile, open(output_path, 'w', newline='') as outfile:
            csv_writer = csv.writer(outfile)
            
            csv_header = [
                'pci', 'distance_ft', 'jammer_dist_ft', 'dl_cqi', 'dl_ri', 'dl_mcs', 'dl_brate', 'dl_ok', 'dl_nok', 'dl_percent', 'dl_bs',
                'ul_pusch', 'ul_rsrp', 'ul_ri', 'ul_mcs', 'ul_brate', 'ul_ok', 'ul_nok', 'ul_percent', 'ul_bsr', 'ul_ta', 'ul_phr'
            ]
            csv_writer.writerow(csv_header)

            for line in infile:
                if processing_state == 'HEADER':
                    jammer_match = jammer_dist_pattern.search(line)
                    if jammer_match:
                        jammer_distance = jammer_match.group(1)
                        continue

                    ue_match = ue_mapping_pattern.search(line)
                    if ue_match:
                        ue_name, rnti, distance = ue_match.groups()
                        
                        rnti = rnti.strip('()') # Clean up the RNTI value

                        if distance.endswith('.0'):
                            distance = distance[:-2]

                        # MODIFIED: Populate both maps
                        ue_to_distance_map[ue_name] = distance
                        rnti_to_ue_map[rnti] = ue_name
                        continue

                    if 'pci rnti' in line and '|' in line:
                        print("Header parsing complete.")
                        if not ue_to_distance_map:
                            print("\nFATAL ERROR: No UE-to-distance mappings found in header.", file=sys.stderr)
                            sys.exit(1)
                        print(f"Found Jammer Distance: {jammer_distance} ft")
                        print(f"Found UE-to-Distance Mapping: {ue_to_distance_map}")
                        print(f"Found RNTI-to-UE Mapping: {rnti_to_ue_map}") # For debugging
                        print("Switching to DATA processing state.")
                        processing_state = 'DATA'
                        continue
                
                if processing_state == 'DATA':
                    # Skip any repeated header lines in the data section
                    if 'pci rnti' in line or '---' in line:
                        continue
                    
                    line_match = data_line_pattern.match(line.strip())
                    if not line_match:
                        continue

                    try:
                        all_values = list(line_match.groups())
                        
                        # MODIFIED: Perform a two-step lookup to get the correct distance
                        rnti = all_values[1]
                        ue_name = rnti_to_ue_map.get(rnti)
                        
                        if ue_name is None:
                            continue 
                        
                        ue_distance = ue_to_distance_map.get(ue_name)
                        if ue_distance is None:
                            continue 
                        
                        row_data = [
                            all_values[0],    # pci
                            ue_distance,      # distance_ft (Now correctly looked up)
                            jammer_distance,  # jammer_dist_ft
                            all_values[2],    # dl_cqi
                            all_values[3],    # dl_ri
                            all_values[4],    # dl_mcs
                            clean_brate(all_values[5]),    # dl_brate
                            all_values[6],    # dl_ok
                            all_values[7],    # dl_nok
                            all_values[8].rstrip('%'),     # dl_percent
                            all_values[9],    # dl_bs
                            all_values[10],   # ul_pusch
                            all_values[11],   # ul_rsrp
                            all_values[12],   # ul_ri
                            all_values[13],   # ul_mcs
                            clean_brate(all_values[14]),   # ul_brate
                            all_values[15],   # ul_ok
                            all_values[16],   # ul_nok
                            all_values[17].rstrip('%'),    # ul_percent
                            clean_brate(all_values[18]),   # ul_bsr
                            all_values[19].rstrip('n'),    # ul_ta
                            all_values[20],   # ul_phr
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
        description="Parse srsRAN gNB log files into a clean CSV format. Extracts UE distances and jammer distance from header.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_file", help="Path to the input srsRAN log file.")
    parser.add_argument("output_file", help="Path for the output CSV file.")
    
    args = parser.parse_args()
    
    process_log_file(args.input_file, args.output_file)
