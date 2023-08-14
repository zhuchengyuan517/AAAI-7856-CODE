import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_specgram(signal, title, output_file):
    fig, ax = plt.subplots()
    ax.specgram(signal, Fs=2000) # Fs is the sampling rate, set to 2000

    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
#   ax.set_title(title)


#    plt.savefig(output_file)
    plt.axis('off')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)

 def process_file(input_path, output_path):
    df = pd.read_csv(input_path)
    signal = df.iloc[:,0].tolist()
#    signal = df['Signal'].values
    output_file = os.path.join(output_path, os.path.basename(input_path).replace('.csv', '.png'))
    create_specgram(signal, os.path.basename(input_path), output_file)

 def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 处理每个csv文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_dir, filename)
            process_file(input_path, output_dir)

input_dir = 'D:/'
output_dir = 'D:/'
process_directory(input_dir, output_dir)