import logging
from tensorflow import keras
from . import config


def model_gen_t1(l1filters=64,
                 l1kernel_size=(35, 35),
                 l1strides=(13, 13),
                 l2filters=64,
                 l2kernel_size=(35, 35),
                 l2strides=(5, 5),
                 d1width=1000,
                 d2width=100,
                 d3width=20,
                 do1=0.05,
                 do2=0.005,
                 do3=0.001):

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


def model_gen_t2(l1filters=16,
                 l1kernel_size=(1, 35),
                 l1strides=(3, 10),
                 pFilters=12,
                 pSizes=[16, 32, 64, 128, 256],
                 pStrides=(3, 10),
                 d1width=1024,
                 d2width=256,
                 d3width=64,
                 do1=0.5,
                 do2=0.05,
                 do3=0.01):

    input_img = keras.layers.Input(
        shape=(config.input_freq_dim, config.input_time_dim, 1))
    # Initial convolutional layers, 'cos why not.
    conv = keras.layers.Conv2D(
        l1filters, l1kernel_size, l1strides, padding='same', activation='elu')(input_img)
    conv = keras.layers.Dropout(0.5)(conv)

    conv = keras.layers.Conv2D(
        l1filters, l1kernel_size, l1strides, padding='same', activation='elu')(conv)
    conv = keras.layers.Dropout(0.5)(conv)

    # Run a few parallel conv layers to try and figure out spacing.
    def pconv(size, input_l):
        c = keras.layers.Conv2D(
            pFilters, (1, size), pStrides, padding='same', activation='elu')(input_l)
        return keras.layers.Dropout(0.5)(c)

    parallel = [pconv(s, conv) for s in pSizes]

    concat = keras.layers.Concatenate()(parallel)

    flat = keras.layers.Flatten()(concat)

    dense = keras.layers.Dense(d1width, activation='elu')(flat)
    dense = keras.layers.Dropout(do1)(dense)

    dense = keras.layers.Dense(d2width, activation='elu')(dense)
    dense = keras.layers.Dropout(do2)(dense)

    dense = keras.layers.Dense(d3width, activation='elu')(dense)
    dense = keras.layers.Dropout(do3)(dense)

    output = keras.layers.Dense(1)(dense)

    return keras.Model(inputs=input_img, outputs=output)


models = {
    "v1": lambda: model_gen_t1(),
    "v2": lambda: model_gen_t1(l1kernel_size=(5, 5), l1strides=(3, 3), l2strides=(13, 13)),
    "v3": lambda: model_gen_t1(l1strides=(3, 3), l2strides=(13, 13), d1width=4096, d2width=128, d3width=32),
    "v4": lambda: model_gen_t1(l2strides=(3, 3), d1width=4096, d2width=128, d3width=32),
    "v5": lambda: model_gen_t1(l2strides=(3, 3), d1width=4096, d2width=512, d3width=512),
    "v6": lambda: model_gen_t1(l1filters=128, l2filters=128, l2strides=(3, 3), d1width=4096, d2width=512, d3width=512),
    "v7": lambda: model_gen_t1(l1filters=16, l2filters=16, l2strides=(3, 3), d1width=2048, d2width=1024, d3width=512),
    "v8": lambda: model_gen_t1(l1filters=32, l1strides=(11, 11), l2filters=32, l2strides=(3, 3), d1width=2048, d2width=1024, d3width=512),
    "v9": lambda: model_gen_t2(),


}


def gen_latest_model():
    return models['v9']()


def load_model(modelfile):
    logging.info(f"Loading model file {modelfile}")
    model = None
    try:
        model = keras.models.load_model(modelfile, compile=False)
        logging.debug("Loaded model directly.")
    except Exception as e:
        logging.error(f"Threw: {str(e)}")
        logging.debug("Manually creating model")
        for name, model_gen in models.items():
            try:
                model = model_gen()
                model.load_weights(modelfile)
                break
            except Exception as e:
                logging.error(f"Threw: {str(e)}")
                logging.debug(f"Failed to load weights for model {name}")
    return model


def find_best_model_in_directory(directory, metric="epoch"):
    assert metric == "epoch" or metric == "loss"
    import glob
    import re

    name_regex = "model-epoch-(\d+)-loss-(\d+\.\d+).hdf5"

    best_epoch = int(0)
    best_loss = float(1e10)
    best_model_file = None

    for model_file in glob.glob(directory + "/*.hdf5"):
        try:
            m = re.search(name_regex, model_file)
            epoch = int(m.group(1))
            loss = float(m.group(2))
            logging.info(f"Epoch: {epoch}")
            logging.info(f"Loss: {loss}")

            if metric == "epoch" and epoch > best_epoch:
                best_epoch = epoch
                best_loss = loss
                best_model_file = model_file
                logging.info(f"Found newer model file: {model_file}")

            if metric == "loss" and loss < best_loss:
                best_epoch = epoch
                best_loss = loss
                best_model_file = model_file
                logging.info(
                    f"Found model file with lower loss: {model_file}")

        except AttributeError:
            logging.error(
                f"Error extracting epoch/loss from model file name {model_file}")

    return (best_model_file, best_epoch, best_loss)
