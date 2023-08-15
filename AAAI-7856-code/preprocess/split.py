import csv
import os
import pandas as pd

# Root directory of data
dir = 'D:\\Code\\data\\raw data'
files = os.listdir(dir)
for file in files:
    file_path = dir + '/' + file + '/' + file + f'.csv'
    save_path = dir + '/' + file + '/'
    df = pd.read_csv(file_path, header=None).values
    # Open the original csv file
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        data = list(reader)
    # Define the length of each segment
    segment_length = 20000

    # Calculate the number of segments
    num_segments = len(data) // segment_length
    # Loop through each segment and save as a new csv file
    for i in range(num_segments):
        start_index = i * segment_length
        end_index = start_index + segment_length
        segment_data = data[start_index:end_index]
        with open(save_path + file + f'_segment_{i + 1}.csv', 'w', newline='') as f_save:
            writer = csv.writer(f_save)
            writer.writerows(segment_data)
