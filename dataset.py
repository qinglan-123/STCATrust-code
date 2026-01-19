import pandas as pd
import datetime

# Define a function to get dataset snapshot indices by time division
# time_slots: indicates how many time slots to divide the dataset into
# data_path: indicates the path of the dataset file
def get_snapshot_index(time_slots, data_path):
    # Use pandas' read_csv function to read the dataset file at the specified path and store it in DataFrame object df
    df = pd.read_csv(data_path)
    # Specify column names for the DataFrame to facilitate subsequent operations
    df.columns = ['source','target','rating','time']
    # Extract time column data from the DataFrame and convert it to a list
    time_list = list(df['time'])
    # Sort the time list to ensure time is in ascending order
    time_list.sort()

    # Get the first timestamp in the time list, representing the start time of the dataset
    ts_begin = time_list[0]
    # Convert the start timestamp to a date-time string in the specified format
    begin = datetime.datetime.fromtimestamp(ts_begin).strftime('%Y-%m-%d %H:%M:%S')
    # Get the last timestamp in the time list, representing the end time of the dataset
    ts_finish = time_list[-1]
    # Convert the end timestamp to a date-time string in the specified format
    finish = datetime.datetime.fromtimestamp(ts_finish).strftime('%Y-%m-%d %H:%M:%S')
    # Print the total number of interactions in the dataset and the start and end times
    print('Total {} interactions from {} to {}'.format(len(time_list),begin,finish))
    # Calculate the total number of days in the dataset, i.e., the difference in days between end time and start time
    days = (datetime.datetime.fromtimestamp(ts_finish)-datetime.datetime.fromtimestamp(ts_begin)).days
    # Print the total number of days in the dataset
    print('Total {} days'.format(days))

    # Calculate the time span, i.e., the difference between end timestamp and start timestamp
    span = ts_finish-ts_begin
    # Initialize an empty list to store the split timestamps for each time slot
    split_list = []
    # Loop to generate split timestamps for all but the last time slot
    for i in range(1,time_slots):
        # Calculate the split timestamp for the current time slot by adding the average time interval to the start timestamp
        ts_begin += (span//time_slots)
        # Add the calculated split timestamp to the split_list
        split_list.append(ts_begin)
    # Add the end timestamp to the split_list as the split point for the last time slot
    split_list.append(ts_finish)

    # Initialize an empty list to store the dataset indices corresponding to each time slot
    index_list = []
    # Iterate through each split timestamp in split_list
    for span in split_list:
        # If the current split timestamp is the last split timestamp
        if span == split_list[-1]:
            # Add the last index of the dataset to index_list
            index_list.append(len(time_list)-1)
            # Break the loop since the last time slot has been processed
            break
        # Iterate through each timestamp in the time list
        for j in range(len(time_list)):
            # If the current timestamp is less than the split timestamp and the next timestamp is greater than the split timestamp
            if time_list[j] < span < time_list[j + 1]:
                # Add the current index to index_list, representing the end index of the time slot
                index_list.append(j)

    # Iterate through each split timestamp in split_list to print information for each time slot
    for i in range(len(split_list)):
        # If it's the first time slot
        if i == 0:
            # Print information for the first time slot, including time slot number, time, and number of edges
            print('In snapshot {}'.format(i),'Time:{}'.format(datetime.datetime.fromtimestamp(split_list[i]).strftime('%Y-%m-%d %H:%M:%S')),'#Edges={}'.format(
            index_list[i]))
        else:
            # Print information for other time slots, including time slot number, time, and number of edges (difference between current and previous time slot)
            print('In snapshot {}'.format(i),'Time:{}'.format(datetime.datetime.fromtimestamp(split_list[i]).strftime('%Y-%m-%d %H:%M:%S')),'#Edges={}'.format(
            index_list[i] - index_list[i-1]))
    # Return the list of dataset indices corresponding to each time slot
    return index_list