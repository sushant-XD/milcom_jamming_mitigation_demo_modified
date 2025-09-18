#!/bin/bash
if [ $EUID -ne 0 ]; then
  echo "script must run as root"
  exit 1
fi

echo "Running 5G Core as $SUDO_USER ..."
cd /home/$SUDO_USER/srsRAN_Project/docker
docker compose up 5gc
echo "Shutting Down"
exit
