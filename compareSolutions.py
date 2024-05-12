import os
import pandas as pd
import numpy as np
import heartpy as hp

from buildModel import loadE2EModel

if __name__ == "__main__":
    model = loadE2EModel("models/e2e_model.h5")
    model.summary()

    df = pd.read_csv('dataset.csv')
    df['window'] = df['window'].apply(lambda x: np.fromstring(x.replace('[', '').replace(']', ''), sep=','))

    X = np.stack(df['window'].values)
    X = np.expand_dims(X, axis=-1)
    #TODO process x sample files and compare heartpy solution against the neuron solution
