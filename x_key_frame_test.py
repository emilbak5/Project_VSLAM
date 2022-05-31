import json
import numpy as np
import os

import matplotlib.pyplot as plt



# make a list of number with values from 10 to 300 in jumps of 10
threshholds = [i for i in range(10, 610, 10)]

mean_errors = []
sum_times = []
length_errors = []

file_names = []

for value in threshholds:
    file_names.append("keyframe_test" + str(value) + ".json")







# iterate through all the json file in the folder keyframe_test
for i, file_name in enumerate(file_names):

    # read the data.json and values in lists
    with open('./keyframe_test/' + file_name) as json_file:
        data = json.load(json_file)
        error = data['error']
        time = data['time']



    # calculate the mean and max of the error
    mean_error = np.mean(error)
    length_error = len(error)

    # calculate the mean and max of the time
    sum_time = np.sum(time) / 1000

    # append the mean and max of the error and time to the list
    mean_errors.append(mean_error)
    sum_times.append(sum_time)
    length_errors.append(length_error)






# delete the first element of the list
mean_errors.pop(0)
sum_times.pop(0)
length_errors.pop(0)

mean_errors.pop(0)
sum_times.pop(0)
length_errors.pop(0)

frames_removed = []
for length_error in length_errors:
    frames_removed.append(1000 - length_error)
print(frames_removed)

# only keep every 5th element in frames_removed
frames_removed = frames_removed[::5]
print(threshholds[2::5])
print(frames_removed)


# plot the mean_errors and threshold

plt.plot(threshholds[0:len(mean_errors)], mean_errors)
plt.xlabel('Threshold')
plt.ylabel('Sum Error [m]')
plt.title('Mean Error vs Threshold')
# save the plot
plt.savefig('sum_error_vs_threshold.png')
plt.show()

# clear the plot
plt.clf()
# make a plot of the sum_times and threshold
plt.plot(threshholds[1:len(sum_times)+1], sum_times)
plt.xlabel('Threshold')
plt.ylabel('Sum Time [ms]')
plt.title('Total Time vs Threshold')
# save the plot
plt.savefig('sum_time_vs_threshold.png')
plt.show()

# make a plot of the length_error
plt.clf()
plt.plot(threshholds[1:len(length_errors)+1], length_errors)
plt.xlabel('Threshold')
plt.ylabel('Number of Images')
plt.title('Number of Images vs Threshold')
# save the plot
plt.savefig('length_error_vs_threshold.png')
plt.show()










