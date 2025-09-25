#!/bin/bash

# --- Configuration ---
GNB_EXECUTABLE="gnb" 
GNB_PROCESS_NAME="gnb"

# Config file paths
ANTIJAMMING_GNB_CONFIG="/home/ntia/ran-tester-ue/configs/uhd/gnb_uhd.yaml" 

# --- Functions ---

stop_gnb() {
    echo "Stopping existing gNodeB processes..."
    sudo pkill -f "$GNB_PROCESS_NAME"
    sleep 2
}

start_gnb() {
    CONFIG_FILE=$1
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config file not found at $CONFIG_FILE"
        exit 1
    fi
    echo "Starting gNodeB with config: $CONFIG_FILE"
    gnome-terminal -- /bin/bash -c "sudo $GNB_EXECUTABLE -c $CONFIG_FILE; exec bash"
}

# --- Main Logic ---

COMMAND=$1

case "$COMMAND" in
    restart_jamming)
        stop_gnb
        sleep 6
        start_gnb "$ANTIJAMMING_GNB_CONFIG"
        echo "gNodeB restarted in anti-jamming mode."
        ;;
    stop)
        stop_gnb
        echo "gNodeB stopped."
        ;;
    *)
        echo "Usage: $0 {restart_jamming|restart_normal|stop}"
        exit 1
        ;;
esac
