#!/bin/bash
chmod u+x 5G_Core.sh
gnome-terminal --title="5g_core" -- bash -c 'sudo ./5G_Core.sh;exec bash'
echo "5G Core Running"
sleep 30 
chmod u+x gNodeB.sh
gnome-terminal --title="gNodeB" -- bash -c 'sudo ./gNodeB.sh;exec bash'
echo "gNodeB Running"
sleep 10
gnome-terminal --title="Inference Script" -- bash -c 'source venv/bin/activate; python3 inference.py;exec bash'
echo "Inference Script Running"
sleep 20
chmod u+x gnb.sh
gnome-terminal --title="Iperf" -- bash -c 'sudo ./gnb.sh;exec bash'
echo "Iperf server running"
chmod u+x jammer.sh
gnome-terminal --title="Jammer Terminal" -- bash -c 'sudo ./jammer.sh;exec bash'
echo "Jammer script running"
