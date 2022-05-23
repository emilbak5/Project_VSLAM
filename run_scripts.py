import subprocess



total_string = ''

for i in range(10, 310, 10):
    string_ = '/bin/python3 "/mnt/c/Users/emilb/OneDrive/Documents/Skole/8. semester/SLAM project/Project_VSLAM/src/main.py" --threshhold=' + str(i) + ' ; '
    total_string += string_

# remove the last &
total_string = total_string[:-3]

x=5
subprocess.run(total_string, shell=True)
