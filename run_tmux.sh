#!/usr/bin/env bash
set -xeuo pipefail

USER=${USER:-$(id -un)}

CORE_DOCKER_PATH="/home/$USER/srsRAN_Project/docker"
SCRIPTS_PATH="/home/$USER/milcom_jamming_mitigation_demo_modified"
SCRIPTS_PATH="/home/$USER/milcom_jamming_mitigation_demo_modified"
JAMMER_PATH="/home/$USER/jammer"

TMUX_SESSION="5g"
WINDOW_TITLE="5G Launcher"

TERMINAL="foot"   # default; can be "gnome" to use gnome-terminal

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  -h, --help          Show this help
  -t, --terminal NAME Set terminal program: "foot" (default) or "gnome"
EOF
    exit 0
}

# simple arg parsing
while [[ "${1:-}" != "" ]]; do
    case "$1" in
        -h|--help) usage ;;
        -t|--terminal)
            shift
            TERMINAL="${1:-}"
            if [[ -z "$TERMINAL" ]]; then
                echo "Missing terminal name after -t/--terminal"
                exit 1
            fi
            ;;
        *) echo "Unknown arg: $1"; usage ;;
    esac
    shift
done

# Check dependencies required for tmux and the chosen terminal
required_cmds=(tmux sudo)
if [[ "$TERMINAL" == "foot" ]]; then
    required_cmds+=(foot)
elif [[ "$TERMINAL" == "gnome" ]]; then
    required_cmds+=(gnome-terminal)
else
    echo "Unsupported terminal: $TERMINAL"
    exit 1
fi

for cmd in "${required_cmds[@]}"; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "Error: '$cmd' is required but not installed."
        exit 1
    fi
done

CMD_5GC='sudo docker compose up --build 5gc'
CMD_GNB='script -f gnb_session.log; sudo gnb -c gnb_uhd.yaml'
CMD_IPERF='set -x; sudo ./iperf_server.sh'
CMD_AI='source ./.venv/bin/activate 2>/dev/null && python3 inference.py'
CMD_JAMMER='sudo ./build/jammer --config configs/basic_jammer.yaml'

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "A tmux session named '$TMUX_SESSION' already exists."
    read -r -p "Kill it and recreate? [y/N] " yn
    case "$yn" in
        [Yy]* ) tmux kill-session -t "$TMUX_SESSION" ;;
        * )
            echo "Attaching to existing session..."
            if [[ "$TERMINAL" == "foot" ]]; then
                foot -T "$WINDOW_TITLE" bash -lc "tmux attach -t $TMUX_SESSION"
            else
                gnome-terminal --title="$WINDOW_TITLE" -- bash -lc "tmux attach -t $TMUX_SESSION"
            fi
            exit 0
            ;;
    esac
fi

# Create detached tmux session starting in CORE_DOCKER_PATH (pane 0, top-left)
tmux new-session -d -s "$TMUX_SESSION" -c "$CORE_DOCKER_PATH"

tmux split-window -h -t "$TMUX_SESSION":0 -c "$SCRIPTS_PATH"

tmux select-pane -t "$TMUX_SESSION":0.0
tmux split-window -v -t "$TMUX_SESSION":0.0 -c "$SCRIPTS_PATH"
tmux split-window -v -t "$TMUX_SESSION":0.1 -c "$SCRIPTS_PATH"

tmux send-keys -t "$TMUX_SESSION":0.0 "$CMD_5GC" C-m
tmux send-keys -t "$TMUX_SESSION":0.1 "$CMD_AI" C-m
tmux send-keys -t "$TMUX_SESSION":0.2 "$CMD_GNB" C-m
tmux send-keys -t "$TMUX_SESSION":0.3 "$CMD_IPERF" C-m

# Ask tmux to balance layout (tiled should produce the 2x2 grid)
tmux select-layout -t "$TMUX_SESSION" tiled || true

# Attach the tmux session inside the requested terminal
if [[ "$TERMINAL" == "foot" ]]; then
    foot -T "$WINDOW_TITLE" bash -lc "tmux attach -t $TMUX_SESSION"
else
    gnome-terminal --title="$WINDOW_TITLE" -- bash -lc "tmux attach -t $TMUX_SESSION"
fi

exit 0

