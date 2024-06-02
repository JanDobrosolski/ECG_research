import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, AdamW
from sklearn.model_selection import train_test_split

def buildModel(autoencoder, encoder_output):
    x = tf.keras.layers.Flatten(name="flatten_rmssd")(encoder_output)
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output_rrsd = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs=autoencoder.input, outputs=output_rrsd)

    return model

if __name__ == "__main__":
    modelsDir = "models"
    autoencoderPath = os.path.join(modelsDir, "autoencoder.h5")
    if not os.path.exists(autoencoderPath):
        raise Exception("Autoencoder model not found, it has to be trained first")
    
    # Load the dataset
    df = pd.read_csv('dataset.csv')

    checkpointDir = "tmp/checkpoint"

    os.makedirs(checkpointDir, exist_ok=True)

    # Convert 'window' from strings of numbers to numpy arrays
    df['window'] = df['window'].apply(lambda x: np.fromstring(x.replace('[', '').replace(']', ''), sep=','))

    df.drop(df[df['rmssdFound'] == 0].index, inplace=True)
    # df.drop(df[df['rmssd'] > np.percentile(df['rmssd'], 95)].index, inplace=True)
    # df.drop(df[df['rmssd'] < np.percentile(df['rmssd'], 5)].index, inplace=True)
    df = df.sample(n=len(df)).reset_index(drop=True)

    X = np.stack(df['window'].values)
    X = np.expand_dims(X, axis=-1)

    X_train, X_temp, y_train, y_temp = train_test_split(X, df['rmssd'].to_numpy(), test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    autoencoder = tf.keras.models.load_model(autoencoderPath)
    encoder_output = autoencoder.get_layer('max_pooling1d_1').output
    encoder_model = tf.keras.Model(inputs=autoencoder.input, outputs=encoder_output)

    for layer in encoder_model.layers:
        layer.trainable = False

    input_shape = X_train.shape[1:]
    input_layer = tf.keras.layers.Input(shape=input_shape)

    model = buildModel(autoencoder, encoder_output) 
    model.compile(optimizer=AdamW(learning_rate=0.001), loss='mse', metrics=['mae'])


    checkpoint_name = "rmssd_checkpoint"
    model_name = "rmssd.h5"
    metrics_name = "rmssd_metric.txt"

    # Setup model checkpoint
    checkpoint_filepath = os.path.join(checkpointDir, checkpoint_name)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_mae',
        mode='min',
        save_best_only=True)

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=128, callbacks=[model_checkpoint_callback])

    model.load_weights(checkpoint_filepath)

    # Evaluate the model
    test_results = model.evaluate(X_test, y_test)
    test_results_path = os.path.join(modelsDir, metrics_name)
    print("Test Results - Loss: ", test_results)

    if os.path.exists(test_results_path):
        with open(test_results_path, 'r') as f:
            old_test_results = float(f.read())
    else:
        old_test_results = 9999.999

    if test_results[1] < old_test_results:
        with open(test_results_path, 'w') as f:
            f.write(str(test_results[1]))


        model.save(os.path.join(modelsDir, model_name))
        # _ = os.popen("python buildModel.py").read()
        