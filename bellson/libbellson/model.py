from tensorflow import keras
import logging
from . import config


def model_gen(l1filters=64, l1kernel_size=(35, 35), l1strides=(13, 13), l2filters=64, l2kernel_size=(35, 35), l2strides=(5, 5),  d1width=1000, d2width=100, d3width=20, do1=0.05, do2=0.005, do3=0.001):
    input_img = keras.layers.Input(
        shape=(config.input_freq_dim, config.input_time_dim, 1))

    base = keras.layers.Conv2D(
        l1filters, l1kernel_size, l1strides, padding='same', activation='relu')(input_img)

    base = keras.layers.Conv2D(
        l2filters, l2kernel_size, l2strides, padding='same', activation='relu')(base)

    flat = keras.layers.Flatten()(base)

    dense = keras.layers.Dense(d1width, activation='relu')(flat)
    dense = keras.layers.Dropout(do1)(dense)

    dense = keras.layers.Dense(d2width, activation='relu')(dense)
    dense = keras.layers.Dropout(do2)(dense)

    dense = keras.layers.Dense(d3width, activation='relu')(dense)
    dense = keras.layers.Dropout(do3)(dense)

    output = keras.layers.Dense(1)(dense)

    return keras.Model(inputs=input_img, outputs=output)


models = {
    "v1": lambda: model_gen(),
    "v2": lambda: model_gen(l1kernel_size=(5, 5), l1strides=(3, 3), l2strides=(13, 13)),
    "v3": lambda: model_gen(l1strides=(3, 3), l2strides=(13, 13), d1width=4096, d2width=128, d3width=32),
    "v4": lambda: model_gen(l2strides=(3, 3), d1width=4096, d2width=128, d3width=32),
    "v5": lambda: model_gen(l2strides=(3, 3), d1width=4096, d2width=512, d3width=512),
    "v6": lambda: model_gen(l1filters=128, l2filters=128, l2strides=(3, 3), d1width=4096, d2width=512, d3width=512),
    "v7": lambda: model_gen(l1filters=16, l2filters=16, l2strides=(3, 3), d1width=2048, d2width=1024, d3width=512),
}


def gen_latest_model():
    return models['v7']()


def load_model(modelfile):
    model = None
    try:
        model = keras.models.load_model(modelfile, compile=False)
    except Exception as e:
        logging.error(f"Threw: {str(e)}")
        logging.info("Manually creating model")
        model = model_gen()
        model.load_weights(modelfile)
    return model
