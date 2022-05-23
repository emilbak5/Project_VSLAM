from dataclasses import dataclass
import json
import numpy as np



# read the data.json and values in lists
with open('data.json') as json_file:
    data = json.load(json_file)
    error = data['error']
    time = data['time']



# calculate the mean and max of the error
mean_error = np.mean(error)
max_error = np.max(error)

# calculate the mean and max of the time
mean_time = np.mean(time)
max_time = np.max(time)


#print all the information
print('mean error: ', mean_error)
print('max error: ', max_error)
print('mean time: ', mean_time)
print('max time: ', max_time)



