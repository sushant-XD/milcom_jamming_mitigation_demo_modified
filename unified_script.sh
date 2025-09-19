#!/bin/bash

set -x 
# All-in-one tmux 5G network launcher

SESSION_NAME="5g"
SCRIPTS_PATH="${1:-.}"

# Function to handle command line options
case "$1" in
    "attach"|"a")
        tmux attach-session -t "$SESSION_NAME" 2>/dev/null || echo "No session found"
        exit 0
        ;;
    "kill"|"k")
        tmux kill-session -t "$SESSION_NAME" 2>/dev/null && echo "Session killed"
        exit 0
        ;;
    "list"|"l")
        tmux list-sessions 2>/dev/null || echo "No sessions"
        exit 0
        ;;
    "-h"|"--help"|"help")
        echo "Usage: $0 [path|attach|kill|list]"
        echo "  path   - launch with scripts in specified path"
        echo "  attach - attach to existing session"
        echo "  kill   - kill session"
        echo "  list   - list sessions"
        exit 0
        ;;
esac

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux not installed"
    echo "Install: sudo pacman -S tmux (Arch) or sudo apt install tmux (Ubuntu)"
    exit 1
fi

# Check if scripts directory exists
if [ ! -d "$SCRIPTS_PATH" ]; then
    echo "Error: Directory $SCRIPTS_PATH not found"
    exit 1
fi

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

echo "Starting 5G network in tmux session: $SESSION_NAME"
echo "Scripts path: $SCRIPTS_PATH"

# Create new session and setup panes
tmux new-session -d -s "$SESSION_NAME" -c "$SCRIPTS_PATH"
tmux split-window -h
tmux split-window -v -t 0
tmux split-window -v -t 1
tmux split-window -v -t 3

# Attach to session
tmux attach-session -t "$SESSION_NAME"

# Define the embedded scripts as functions

# 5G Core function
run_5g_core() {
if [ $EUID -ne 0 ]; then
  echo "script must run as root"
  exit 1
fi
echo "Running 5G Core as $SUDO_USER ..."
cd /home/$SUDO_USER/srsRAN_Project/docker
docker compose up 5gc
echo "Shutting Down"

}

# gNodeB function  
run_gnodeb() {
  echo "Starting gNodeB"
  cd /home/$SUDO_USER/ran-tester-ue
  script -f -c "sudo gnb -c configs/uhd/gnb_uhd.yaml" gnb_session.log
  echo "gNodeB script finished."
}

# Start components in each pane
# Pane 0: 5G Core
tmux send-keys -t 0 'run_5g_core' Enter

sleep 30

# Pane 1: gNodeB
tmux send-keys -t 1 'run_gnodeb' Enter

sleep 10

# Pane 2: AI Inference
tmux send-keys -t 2 '
echo "Starting AI Inference"
if [ -f "inference.py" ]; then
    source venv/bin/activate 2>/dev/null || true
    python3 inference.py
else
    echo "inference.py not found"
fi
' Enter

sleep 20

# Pane 3: Iperf/gnb
tmux send-keys -t 3 '
echo "Starting Iperf server"
# Add your iperf/gnb commands here
echo "Iperf server running"
' Enter

sleep 5

# Pane 4: Jammer
tmux send-keys -t 4 '
echo "Starting Jammer"
# Add your jammer commands here
echo "Jammer running"
' Enter

echo "All components started. Attaching to session..."
echo "Use Ctrl+b d to detach, '$0 attach' to reattach"


