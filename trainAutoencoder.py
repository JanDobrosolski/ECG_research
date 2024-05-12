import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('dataset.csv')

    checkpointDir = "tmp/checkpoint"
    modelsDir = "models"

    os.makedirs(checkpointDir, exist_ok=True)
    os.makedirs(modelsDir, exist_ok=True)

    # Convert 'window' from strings of numbers to numpy arrays
    df['window'] = df['window'].apply(lambda x: np.fromstring(x.replace('[', '').replace(']', ''), sep=','))

    df.drop(df[df['rmssdFound'] == 0].index, inplace=True)
    # df.drop(df[df['rmssd'] > np.percentile(df['rmssd'], 95)].index, inplace=True)
    # df.drop(df[df['rmssd'] < np.percentile(df['rmssd'], 5)].index, inplace=True)

    X = np.stack(df['window'].values)
    X = np.expand_dims(X, axis=-1)

    X_train, X_temp = train_test_split(X, test_size=0.3, random_state=42)
    X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

    input_shape = X_train.shape[1:]
    input_layer = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(x)

    # Decoder
    x = tf.keras.layers.Conv1DTranspose(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling1D(2)(x)
    output_layer = tf.keras.layers.Conv1D(filters=1, kernel_size=3, activation='sigmoid', padding='same')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')

    # Setup model checkpoint
    checkpoint_filepath = os.path.join(checkpointDir, 'autoencoder_checkpoint')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    # Train the model
    model.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=20, batch_size=100, callbacks=[model_checkpoint_callback])

    # Evaluate the model
    test_results = model.evaluate(X_test, X_test)
    print("Test Results - Loss: ", test_results)

    # Optionally, save the model
    model.save(os.path.join(modelsDir,'autoencoder.h5'))
