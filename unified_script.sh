#!/bin/bash

# All-in-one gnome-terminal 5G network launcher

RAN_TESTER_UE_PATH="/home/$USER/ran-tester-ue/"
CORE_DOCKER_PATH="/home/$USER/srsRAN_Project/docker/"
AI_INFERENCE_PATH="/home/$USER/milcom_jamming_mitigation_demo_modified/"
IPERF_SCRIPT_PATH="/home/$USER/milcom_jamming_mitigation_demo_modified/"
JAMMER_PATH="/home/$USER/jammer/"

# Display help message
if [[ "$1" == "-h" || "$1" == "--help" || "$1" == "help" ]]; then
    echo "Usage: $0 [path]"
    echo "  path - launch with scripts in the specified path"
    echo "         (defaults to the current directory)"
    exit 0
fi

# Check if gnome-terminal is installed
if ! command -v gnome-terminal &> /dev/null; then
    echo "Error: gnome-terminal not found."
    exit 1
fi


echo "Starting 5G network in gnome-terminal..."
echo "Starting as $USER"
# Tab 1: 5G Core
#CMD_5GC="echo \"Running 5G Core...\"; cd $CORE_DOCKER_PATH;sudo docker compose up 5gc; echo \"5GC process finished.\"; exec bash"
#gnome-terminal --tab --title="5G Core" -- bash -c "$CMD_5GC"
#echo "5G core started"
#sleep 30 # Give time for the first tab to initialize

# Tab 2: gNodeB
#CMD_GNB="echo \"Starting gNodeB...\"; cd $RAN_TESTER_UE_PATH; script -f -c \"sudo gnb -c configs/uhd/gnb_uhd.yaml\" gnb_session.log;echo \"gNodeB process finished.\"; exec bash"
#gnome-terminal --tab --title="gNodeB" -- bash -c "$CMD_GNB"
#echo "gNodeB running"
#sleep 20
# Tab 4: Iperf
CMD_IPERF="echo \"Iperf server starting...\"; cd $IPERF_SCRIPT_PATH;sudo ./gnb.sh; echo \"Iperf ready.\"; exec bash"
gnome-terminal --tab --working-directory="$SCRIPTS_PATH" --title="Iperf" -- bash -c "$CMD_IPERF"
sleep 10

# Tab 3: AI Inference
CMD_AI="echo \"Starting AI Inference...\"; cd $AI_INFERENCE_PATH;source venv/bin/activate;python3 inference.py;echo \"AI process finished.\"; exec bash"
gnome-terminal --tab --working-directory="$SCRIPTS_PATH" --title="AI Inference" -- bash -c "$CMD_AI"
echo "AI inference running"
sleep 10

echo "All other components are starting in separate tabs."
echo "--> To start the jammer, type 'j' or 'jammer' and press Enter."

# Loop to wait for user input to start the jammer
while true; do
    read -p "Start jammer? (j/jammer): " user_input
    if [[ "$user_input" == "j" || "$user_input" == "jammer" ]]; then
        # Tab 5: Jammer
        CMD_JAMMER="echo \"Jammer starting...\"; cd $JAMMER_PATH;sudo ./build/jammer --config configs/basic_jammer.yaml;echo \"Jammer process finished.\"; exec bash"
        gnome-terminal --tab --working-directory="$JAMMER_PATH" --title="Jammer" -- bash -c "$CMD_JAMMER"
        echo "Jammer is starting."
        break
    else
        echo "Invalid input. Please enter 'j' or 'jammer' to start."
    fi
done

echo "Launcher script has finished."
