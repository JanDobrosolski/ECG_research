import os
import tensorflow as tf
import pandas as pd
import numpy as np

class ConditionalActivationLayer(tf.keras.layers.Layer):

    def __init__(self, f1, f2, **kwargs):
        super(ConditionalActivationLayer, self).__init__(**kwargs)
        self.f1 = f1
        self.f2 = f2

    def get_config(self):
        config = super().get_config()
        config.update({
            'f1': self.f1,
            'f2': self.f2
        })
        return config

    def call(self, inputs):
        x = inputs
        x = self.f1(x)
        condition = tf.greater(x, 0.5)
        return tf.where(condition, self.f2(inputs), tf.zeros_like(x))

def loadE2EModel(model_path):
    custom_objects = {"ConditionalActivationLayer": ConditionalActivationLayer}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

if __name__ == "__main__":
    modelsDir = "models"
    autoencoderPath = os.path.join(modelsDir, "autoencoder.h5")
    rmssdpath = os.path.join(modelsDir, "rmssd.h5")
    discriminatorPath = os.path.join(modelsDir, "discriminator.h5")

    if not os.path.exists(autoencoderPath):
        raise Exception("Autoencoder model not found, it has to be trained first")
    
    if not os.path.exists(rmssdpath):
        raise Exception("RMSSD model not found, it has to be trained first")
    
    if not os.path.exists(discriminatorPath):
        raise Exception("Discriminator model not found, it has to be trained first")
    
    autoencoder = tf.keras.models.load_model(autoencoderPath)
    rmssd = tf.keras.models.load_model(rmssdpath)
    discriminator = tf.keras.models.load_model(discriminatorPath)

    encoder_output = autoencoder.get_layer('max_pooling1d_1').output
    encoder_model = tf.keras.Model(inputs=autoencoder.input, outputs=encoder_output)

    discriminator_input = discriminator.get_layer("flatten_discriminate")
    rmssd_input = rmssd.get_layer("flatten")

    discriminator_model = tf.keras.Model(inputs=discriminator_input.input, outputs=discriminator.output)

    rmssd_model = tf.keras.Model(inputs=rmssd_input.input, outputs=rmssd.output)

    outputLayer = ConditionalActivationLayer(discriminator_model, rmssd_model)(encoder_output)

    e2e_model = tf.keras.Model(inputs=autoencoder.input, outputs=outputLayer)
    e2e_model.save(os.path.join(modelsDir, "e2e_model.h5"))
