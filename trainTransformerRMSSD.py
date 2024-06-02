import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class TransformerEncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training):
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

def create_transformer_model(seq_len, input_dim, d_model, num_heads, dff, num_layers, rate=0.1):
    inputs = tf.keras.Input(shape=(seq_len, input_dim))
    
    # Project the input to the model dimension
    x = Dense(d_model)(inputs)

    for _ in range(num_layers):
        x = TransformerEncoderLayer(d_model, num_heads, dff, rate)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(dff, activation='relu')(x)
    x = tf.keras.layers.Dropout(rate)(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def loadTransModel(model_path):
    custom_objects = {"TransformerEncoderLayer": TransformerEncoderLayer}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
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

    # df.drop(df[df['rmssdFound'] == 0].index, inplace=True)
    # df.drop(df[df['rmssd'] > np.percentile(df['rmssd'], 95)].index, inplace=True)
    # df.drop(df[df['rmssd'] < np.percentile(df['rmssd'], 5)].index, inplace=True)
    df = df.sample(n=len(df)).reset_index(drop=True)

    X = np.stack(df['window'].values)
    X = np.expand_dims(X, axis=-1)

    X_train, X_temp, y_train, y_temp = train_test_split(X, df['rmssd'].to_numpy(), test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    seq_len = 1536
    input_dim = 1
    d_model = 128
    num_heads = 8
    dff = 256
    num_layers = 4
    dropout_rate = 0.1

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # Create and compile the model
        model = create_transformer_model(seq_len, input_dim, d_model, num_heads, dff, num_layers, dropout_rate)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

        model.summary()
        checkpoint_name = "rmssd_trans_checkpoint"
        model_name = "rmssd_trans.h5"
        metrics_name = "rmssd_trans_metric.txt"

        # Setup model checkpoint
        checkpoint_filepath = os.path.join(checkpointDir, checkpoint_name)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_mae',
            mode='min',
            save_best_only=True)

        # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=64, callbacks=[model_checkpoint_callback])

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
