# Project_VSLAM
 Project in VSLAM 2022

 # File structure for data
 The data folder for this project should be named "data" and contain the folder of sequences. This folder should be named "sequences_gray" and contain subfolders called "00" "01" etc. 

# How to install Graph-tools
The following line needs to be added to: /etc/apt/sources.list

    deb [ arch=amd64 ] https://downloads.skewed.de/apt focal main

First make a backup copy

    Sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak

Should be able to use a text editor to do it, but else run

    echo "deb [ arch=amd64 ] https://downloads.skewed.de/apt focal main" | sudo tee -a /etc/apt/sources.list

Then, downloade the public key 
https://keys.openpgp.org/search?q=612DEFB798507F25
to verify the packages using the command

    apt-key adv --keyserver keyserver.ubuntu.com --recv-key 612DEFB798507F25

The command have to be used as SUDO and at the same place as the public key (Probably downloads)

After, run "apt-get update" and then 

    apt-get install python3-graph-tool


For a more indepth explanation, visit
https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions
https://askubuntu.com/questions/197564/how-do-i-add-a-line-to-my-etc-apt-sources-list