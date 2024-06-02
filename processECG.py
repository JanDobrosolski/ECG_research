import os
import wfdb
import heartpy as hp
import numpy as np
import pandas as pd
import re

from tqdm import tqdm

WINDOW_STEP = 32
WINDOW_SIZE_S = 3

if __name__ == "__main__":
    ecgList = os.listdir('electrocardiograms')

    window_size = None

    data_to_save = []   

    tqdm.write("Processing ECGs")
    for ecg in tqdm(ecgList):
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

            if np.isnan(normalized_window_data).any():
                continue

            try:
                wd, m = hp.process(normalized_window_data, sampling_rate)
            except:
                data_to_save.append([normalized_window_data.tolist(), 0, 0.])

            if np.isnan(m['rmssd']):
                data_to_save.append([normalized_window_data.tolist(), 0, 0.])
            else:
                data_to_save.append([normalized_window_data.tolist(), 1, m['rmssd']])

    data_path = '/home/macierz/s175327/ECG_research/physionetSignals/ecg-id-database-1.0.0'

    personList = [x for x in os.listdir(data_path) if x.startswith("Person")]

    tqdm.write("Processing people")
    for person in tqdm(personList):
        recordList = list(set([x[:-4] for x in os.listdir(os.path.join(data_path, person)) if x.startswith("rec_")]))

        for record in recordList:
            record_path = os.path.join(data_path, person, record)
            record = wfdb.rdrecord(record_path)
            rawSignal = record.p_signal[:,0]

            for start  in range(0, len(rawSignal) - window_size + 1, WINDOW_STEP):
                window = np.array(rawSignal[start:start + window_size])

                if len(window) != window_size:
                    continue

                min_val = window.min()
                max_val = window.max()
                normalized_window_data = (window - min_val) / (max_val - min_val)

                if np.isnan(normalized_window_data).any():
                    continue

                try:
                    wd, m = hp.process(normalized_window_data, 500)
                except:
                    data_to_save.append([normalized_window_data.tolist(), 0, 0.])

                if np.isnan(m['rmssd']):
                    data_to_save.append([normalized_window_data.tolist(), 0, 0.])
                else:
                    data_to_save.append([normalized_window_data.tolist(), 1, m['rmssd']])

    df = pd.DataFrame(data_to_save, columns=['window', 'rmssdFound', 'rmssd'])
    df.to_csv('dataset.csv', index=False)
