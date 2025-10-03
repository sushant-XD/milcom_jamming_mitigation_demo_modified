#!/bin/bash

# --- Configuration ---
GNB_EXECUTABLE="gnb" 
GNB_PROCESS_NAME="gnb"

# Config file paths
ANTIJAMMING_GNB_CONFIG="/home/charles/ran-tester-ue/configs/uhd/gnb_uhd_alt.yaml" 

# --- Functions ---

stop_gnb() {
    echo "Stopping existing gNodeB processes..."
    sudo kill -9 $(ps aux | awk '/gnb/&&!/awk/&&!/script/{print $0}')
    sleep 2
}

start_gnb() {
    CONFIG_FILE=$1
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config file not found at $CONFIG_FILE"
        exit 1
    fi
    echo "Starting gNodeB with config: $CONFIG_FILE"
    tmux send-keys -t "5g":0.2 "sudo gnb -c gnb_uhd_alt.yaml" C-m
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
