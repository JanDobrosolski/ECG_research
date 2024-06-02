import timeit
import numpy as np
import heartpy as hp

from buildModel import loadE2EModel
from trainTransformerRMSSD import loadTransModel

def normalize_data(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)

def prepare_batches(data, batch_size=1536):
    # Determine the number of full batches we can form
    num_full_batches = data.shape[0] // batch_size
    
    # Resize the array to discard the tail that we can't use
    trimmed_data = data[:num_full_batches * batch_size]
    
    # Reshape the data to fit the model input (num_full_batches, 1536, 1)
    reshaped_data = trimmed_data.reshape(-1, batch_size, 1)
    
    return reshaped_data

if __name__ == "__main__":
    model = loadE2EModel("models/e2e_model.h5")
    # model = loadTransModel("models/rmssd_trans.h5")
    model.summary()


    #quality comparison based on sample data
    classic_results = []
    model_results = []
    model_error = []

    # sample 1
    data, _ = hp.load_exampledata(0)
    data = data[:1536]

    normalized_data = normalize_data(data)

    modelResult = model(np.expand_dims(normalized_data, axis=0))
    wd, m = hp.process(data, 100)

    classic_results.append(m['rmssd'])
    model_results.append(modelResult[0][0].numpy())
    model_error.append(abs(m['rmssd'] - modelResult[0][0].numpy()))

    # sample 2
    data, timer = hp.load_exampledata(1)
    sample_rate = hp.get_samplerate_mstimer(timer)

    wd, m = hp.process(data, sample_rate)

    signal2Results = []
    for i, start in enumerate(range(0, len(data) - 256 + 1, 256)):
        # Extract a window of data from the current position
        window = data[start:start + model.input.shape[1]]

        if window.shape[0] != model.input.shape[1]:
            continue

        normWindow = normalize_data(window)
        modelResult = model(np.expand_dims(normWindow, axis=0))
        if (modelResult.numpy() != 0).all():
            signal2Results.append(modelResult)

    classic_results.append(m['rmssd'])
    model_results.append(np.mean(signal2Results))
    model_error.append(abs(m['rmssd'] - np.mean(signal2Results)))

    #sample 3
    data, timer = hp.load_exampledata(2)
    sample_rate = hp.get_samplerate_datetime(timer, timeformat = '%Y-%m-%d %H:%M:%S.%f')
    wd, m = hp.process(data, sample_rate)

    signal3Results = []
    for i, start in enumerate(range(0, len(data) - 256 + 1, 256)):
        # Extract a window of data from the current position
        window = data[start:start + model.input.shape[1]]

        if window.shape[0] != model.input.shape[1]:
            continue

        normWindow = normalize_data(window)
        modelResult = model(np.expand_dims(normWindow, axis=0))
        if (modelResult.numpy() != 0).all():
            signal3Results.append(modelResult)

    classic_results.append(m['rmssd'])
    model_results.append(np.mean(signal3Results))
    model_error.append(abs(m['rmssd'] - np.mean(signal3Results)))

    testWindow = np.expand_dims(normWindow, axis=0)
    setup_code = 'from __main__ import hp, model, normWindow, sample_rate, testWindow'
    stmt1 = 'hp.process(normWindow, sample_rate)'
    stmt2 = 'model(testWindow)'

    # Measuring the time for the first statement
    time1 = timeit.timeit(stmt=stmt1, setup=setup_code, number=1000)
    print(f"classical processing took {time1:.6f} seconds")

    # Measuring the time for the second statement
    time2 = timeit.timeit(stmt=stmt2, setup=setup_code, number=1000)
    print(f"neuron-based processing took {time2:.6f} seconds")

    # Calculate the speedup
    speedup = time1 / time2
    print(f"Speedup: {speedup:.2f}")

    print("Classic RMSSD: ", classic_results)
    print("Model RMSSD: ", model_results)
    print("Error: ", model_error)

    #save the comparison results
    with open("rmssd_comparison.txt", "w") as f:
        f.write("Classic RMSSD: " + str(classic_results) + "\n")
        f.write("Model RMSSD: " + str(model_results) + "\n")
        f.write("Error: " + str(model_error) + "\n")
        f.write("Speedup: " + str(speedup) + "\n")
        f.write("classic approach time [s]: " + str(time1) + "\n")
        f.write("neuron approach time [s]: " + str(time2) + "\n")
