#!/bin/bash

if [ $EUID -ne 0 ]; then
	echo "Script should be run as root"
	exit 1
fi

set -x

ip ro add 10.45.0.0/16 via 10.53.1.2

iperf3 -s -i 1
