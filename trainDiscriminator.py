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

    impossible_count = len(df[df['rmssdFound']==0])
    possible_count = len(df[df['rmssdFound']==1])

    if impossible_count < possible_count:
        sampled_ok_rows = df[df['rmssdFound']==1].sample(impossible_count)
        df = pd.concat([sampled_ok_rows,df[df['rmssdFound']==0]], ignore_index=True)
    else:
        sampled_non_ok_rows = df[df['rmssdFound']==1].sample(possible_count)
        df = pd.concat([sampled_non_ok_rows,df[df['rmssdFound']==1]], ignore_index=True)

    df = df.sample(frac=1).reset_index(drop=True)

    checkpointDir = "tmp/checkpoint"

    os.makedirs(checkpointDir, exist_ok=True)

    # Convert 'window' from strings of numbers to numpy arrays
    df['window'] = df['window'].apply(lambda x: np.fromstring(x.replace('[', '').replace(']', ''), sep=','))

    X = np.stack(df['window'].values)
    X = np.expand_dims(X, axis=-1)

    X_train, X_temp, y_train, y_temp = train_test_split(X, df['rmssdFound'], test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    autoencoder = tf.keras.models.load_model(autoencoderPath)
    encoder_output = autoencoder.get_layer('max_pooling1d_1').output
    encoder_model = tf.keras.Model(inputs=autoencoder.input, outputs=encoder_output)

    input_shape = X_train.shape[1:]
    input_layer = tf.keras.layers.Input(shape=input_shape)

    # Regression branch converted to classification branch
    x = tf.keras.layers.Flatten(name='flatten_discriminate')(encoder_output)
    x = tf.keras.layers.Dense(128, activation='relu', name='dense_discriminate')(x)
    x = tf.keras.layers.Dropout(0.5, name='dropout_discriminate')(x)
    output_rrsd = tf.keras.layers.Dense(1, activation='sigmoid', name='output_discriminate')(x)
    
    model = tf.keras.Model(inputs=autoencoder.input, outputs=output_rrsd)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Setup model checkpoint
    checkpoint_filepath = os.path.join(checkpointDir, 'discriminator_checkpoint')
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
    print("Test Results - Accuracy: ", test_results[1])

    # Optionally, save the model
    model.save(os.path.join(modelsDir,'discriminator.h5'))
