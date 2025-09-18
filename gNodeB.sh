#!/bin/bash
echo "Starting gNodeB"
cd /home/$SUDO_USER/ran-tester-ue
# The -c flag tells 'script' to run the command instead of an interactive shell
script -f -c "sudo gnb -c configs/uhd/gnb_uhd.yaml" gnb_session.log
echo "gNodeB script finished."
exit
