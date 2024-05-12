import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

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

    X = np.stack(df['window'].values)
    X = np.expand_dims(X, axis=-1)

    X_train, X_temp, y_train, y_temp = train_test_split(X, df['rmssd'].to_numpy(), test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    autoencoder = tf.keras.models.load_model(autoencoderPath)
    encoder_output = autoencoder.get_layer('max_pooling1d_1').output
    encoder_model = tf.keras.Model(inputs=autoencoder.input, outputs=encoder_output)

    input_shape = X_train.shape[1:]
    input_layer = tf.keras.layers.Input(shape=input_shape)

    # Regression branch
    x = tf.keras.layers.Flatten()(encoder_output)  # Flatten the output of the encoder
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)  # Helps prevent overfitting
    output_rrsd = tf.keras.layers.Dense(1, activation='linear')(x)  # Linear activation for regression

    # Full model: input to the autoencoder, output from the regression branch
    model = tf.keras.Model(inputs=autoencoder.input, outputs=output_rrsd)
    model.compile(optimizer='adam', loss='mse', metrics=['mae']) 

    # Setup model checkpoint
    checkpoint_filepath = os.path.join(checkpointDir, 'rmssd_checkpoint')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=100, callbacks=[model_checkpoint_callback])

    # Evaluate the model
    test_results = model.evaluate(X_test, y_test)
    print("Test Results - Loss: ", test_results)

    # Optionally, save the model
    model.save(os.path.join(modelsDir,'rmssd.h5'))
