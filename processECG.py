import os
import heartpy as hp
import numpy as np
import pandas as pd
import re

WINDOW_STEP = 5
WINDOW_SIZE_S = 3

if __name__ == "__main__":
    ecgList = os.listdir('electrocardiograms')

    window_size = None

    data_to_save = []   

    for ecg in ecgList:
        with open(os.path.join('electrocardiograms', ecg), 'r') as f:
            data = f.read().split('\n')

        sampling_rate = int(re.search(r'-?\d+\.?\d*', data[7]).group())

        if window_size is None:
            window_size = sampling_rate*WINDOW_SIZE_S

        ecg_signal = [data_point[0] for data_point in [data_point.split(',') for data_point in data[13:]]][:-1]
        ecg_signal = [int(data_point) for data_point in ecg_signal]

        for start  in range(0, len(ecg_signal) - window_size + 1, WINDOW_STEP):
            window = np.array(ecg_signal[start:start + window_size])

            min_val = window.min()
            max_val = window.max()
            normalized_window_data = (window - min_val) / (max_val - min_val)

            try:
                wd, m = hp.process(normalized_window_data, sampling_rate)
            except:
                data_to_save.append([normalized_window_data.tolist(), 0, 0.])

            if np.isnan(m['rmssd']):
                data_to_save.append([normalized_window_data.tolist(), 0, 0.])
            else:
                data_to_save.append([normalized_window_data.tolist(), 1, m['rmssd']])

    df = pd.DataFrame(data_to_save, columns=['window', 'rmssdFound', 'rmssd'])
    df.to_csv('dataset.csv', index=False)
