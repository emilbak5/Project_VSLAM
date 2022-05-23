import json
import numpy as np
import os

import matplotlib.pyplot as plt



# make a list of number with values from 10 to 300 in jumps of 10
threshholds = [i for i in range(10, 310, 10)]

mean_errors = []
sum_times = []
length_errors = []


# iterate through all the json file in the folder keyframe_test
for file in os.listdir('./keyframe_test'):

    # read the data.json and values in lists
    with open('./keyframe_test/' + file) as json_file:
        data = json.load(json_file)
        error = data['error']
        time = data['time']

    # calculate the mean and max of the error
    sum_error = np.sum(error)
    length_error = len(error)

    # calculate the mean and max of the time
    sum_time = np.sum(time)

    # append the mean and max of the error and time to the list
    mean_errors.append(sum_error)
    sum_times.append(sum_time)
    length_errors.append(length_error)

# delete the first element of the list
# mean_errors.pop(0)
# sum_times.pop(0)


# plot the mean_errors and threshold
plt.plot(threshholds[0:len(mean_errors)], mean_errors)
plt.xlabel('Threshold')
plt.ylabel('Sum Error')
plt.title('Sum Error vs Threshold')
# save the plot
plt.savefig('sum_error_vs_threshold.png')
plt.show()

# clear the plot
plt.clf()
# make a plot of the sum_times and threshold
plt.plot(threshholds[0:len(sum_times)], sum_times)
plt.xlabel('Threshold')
plt.ylabel('Sum Time')
plt.title('Sum Time vs Threshold')
# save the plot
plt.savefig('sum_time_vs_threshold.png')
plt.show()

# make a plot of the length_error
plt.clf()
plt.plot(threshholds[0:len(length_errors)], length_errors)
plt.xlabel('Threshold')
plt.ylabel('Length Error')
plt.title('Length Error vs Threshold')
# save the plot
plt.savefig('length_error_vs_threshold.png')
plt.show()









