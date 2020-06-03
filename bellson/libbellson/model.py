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


def model_gen_t2():

    input_img = keras.layers.Input(
        shape=(config.input_freq_dim, config.input_time_dim, 1))
    # Initial convolutional layers, 'cos why not.
    conv = keras.layers.Conv2D(
        16, (5, 35), (1, 5), padding='same', activation='elu')(input_img)

    conv = keras.layers.Conv2D(
        16, (5, 35), (1, 5), padding='same', activation='elu')(conv)

    # Run a few parallel conv layers to try and figure out spacing.
    def spconv(cin):
        def pconv(size, input_l):
            c = keras.layers.Conv2D(
                12, (1, size), (1, 5), padding='same', activation='elu')(input_l)
            return c

        parallel = [pconv(s, cin) for s in [4, 8, 16, 32, 64, 128]]

        return keras.layers.Concatenate()(parallel)

    spacing_conv = spconv(conv)
    spacing_conv = spconv(spacing_conv)
    spacing_conv = spconv(spacing_conv)

    flat = keras.layers.Flatten()(spacing_conv)

    dense = keras.layers.Dense(1024, activation='elu')(flat)
    dense = keras.layers.Dropout(0.05)(dense)

    dense = keras.layers.Dense(256, activation='elu')(dense)
    dense = keras.layers.Dropout(0.02)(dense)

    dense = keras.layers.Dense(64, activation='elu')(dense)
    dense = keras.layers.Dropout(0.01)(dense)

    output = keras.layers.Dense(1)(dense)

    return keras.Model(inputs=input_img, outputs=output)


def model_gen_t3(f1_f=64, f1_sz=(35, 35), f1_st=(15, 15),
                 f2_f=64, f2_sz=(5, 5), f2_st=(1, 1),
                 cf_f=24, cf_s=5
                 ):
    input_img = keras.layers.Input(
        shape=(config.input_freq_dim, config.input_time_dim, 1))
    # Initial convolutional layers, 'cos why not.
    conv = keras.layers.Conv2D(
        f1_f, f1_sz, f1_st, padding='same', activation='relu')(input_img)

    conv = keras.layers.Conv2D(
        f2_f, f2_sz, f2_st, padding='same', activation='relu')(conv)

    # Run a few parallel conv layers to try and figure out spacing.
    def spconv(cin):
        def pconv(size, input_l):
            c = keras.layers.Conv2D(
                cf_f, (1, size), (1, cf_s), padding='same', activation='relu')(input_l)
            return c

        parallel = [pconv(s, cin) for s in [16, 24, 32, 64, 96, 128, 192, 256]]

        return keras.layers.Concatenate()(parallel)

    spacing_conv = spconv(conv)
    spacing_conv = spconv(spacing_conv)
    spacing_conv = spconv(spacing_conv)

    flat = keras.layers.Flatten()(spacing_conv)

    dense = keras.layers.Dense(1024, activation='relu')(flat)
    dense = keras.layers.Dropout(0.05)(dense)

    dense = keras.layers.Dense(256, activation='relu')(dense)
    dense = keras.layers.Dropout(0.02)(dense)

    dense = keras.layers.Dense(64, activation='elu')(dense)
    dense = keras.layers.Dropout(0.01)(dense)

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
    "v10": lambda: model_gen_t3(f1_st=(15, 17)),  # TODO: Try this!
}


def gen_latest_model():
    return models['v10']()


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
