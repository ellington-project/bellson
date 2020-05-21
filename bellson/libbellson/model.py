from tensorflow import keras


def model_gen(input_time_dim, input_freq_dim):
    return model_gen_v5(input_time_dim, input_freq_dim)


def model_gen_v1(input_time_dim, input_freq_dim, l1filters=64, l2filters=64, d1width=1000, d2width=100, d3width=20, do1=0.05, do2=0.005, do3=0.001):
    input_img = keras.layers.Input(shape=(input_freq_dim, input_time_dim, 1))

    base = keras.layers.Conv2D(l1filters, kernel_size=(35, 35), strides=(
        13, 13), padding='same', activation='relu')(input_img)

    base = keras.layers.Conv2D(l2filters, kernel_size=(35, 35), strides=(
        5, 5), padding='same', activation='relu')(base)

    flat = keras.layers.Flatten()(base)

    dense = keras.layers.Dense(d1width, activation='relu')(flat)
    dense = keras.layers.Dropout(do1)(dense)

    dense = keras.layers.Dense(d2width, activation='relu')(dense)
    dense = keras.layers.Dropout(do2)(dense)

    dense = keras.layers.Dense(d3width, activation='relu')(dense)
    dense = keras.layers.Dropout(do3)(dense)

    output = keras.layers.Dense(1)(dense)

    return keras.Model(inputs=input_img, outputs=output)


def model_gen_v2(input_time_dim, input_freq_dim, l1filters=64, l2filters=64, d1width=1000, d2width=100, d3width=20, do1=0.05, do2=0.005, do3=0.001):
    input_img = keras.layers.Input(shape=(input_freq_dim, input_time_dim, 1))

    base = keras.layers.Conv2D(l1filters, kernel_size=(5, 5), strides=(
        3, 3), padding='same', activation='relu')(input_img)

    base = keras.layers.Conv2D(l2filters, kernel_size=(35, 35), strides=(
        13, 13), padding='same', activation='relu')(base)

    flat = keras.layers.Flatten()(base)

    dense = keras.layers.Dense(d1width, activation='relu')(flat)
    dense = keras.layers.Dropout(do1)(dense)

    dense = keras.layers.Dense(d2width, activation='relu')(dense)
    dense = keras.layers.Dropout(do2)(dense)

    dense = keras.layers.Dense(d3width, activation='relu')(dense)
    dense = keras.layers.Dropout(do3)(dense)

    output = keras.layers.Dense(1)(dense)

    return keras.Model(inputs=input_img, outputs=output)


def model_gen_v3(input_time_dim, input_freq_dim, l1filters=64, l2filters=64, d1width=4096, d2width=128, d3width=32, do1=0.05, do2=0.005, do3=0.001):
    input_img = keras.layers.Input(shape=(input_freq_dim, input_time_dim, 1))

    base = keras.layers.Conv2D(l1filters, kernel_size=(35, 35), strides=(
        3, 3), padding='same', activation='relu')(input_img)

    base = keras.layers.Conv2D(l2filters, kernel_size=(35, 35), strides=(
        13, 13), padding='same', activation='relu')(base)

    flat = keras.layers.Flatten()(base)

    dense = keras.layers.Dense(d1width, activation='relu')(flat)
    dense = keras.layers.Dropout(do1)(dense)

    dense = keras.layers.Dense(d2width, activation='relu')(dense)
    dense = keras.layers.Dropout(do2)(dense)

    dense = keras.layers.Dense(d3width, activation='relu')(dense)
    dense = keras.layers.Dropout(do3)(dense)

    output = keras.layers.Dense(1)(dense)

    return keras.Model(inputs=input_img, outputs=output)


def model_gen_v4(input_time_dim, input_freq_dim, l1filters=64, l2filters=64, d1width=4096, d2width=128, d3width=32, do1=0.05, do2=0.005, do3=0.001):
    input_img = keras.layers.Input(shape=(input_freq_dim, input_time_dim, 1))

    base = keras.layers.Conv2D(l1filters, kernel_size=(35, 35), strides=(
        13, 13), padding='same', activation='relu')(input_img)

    base = keras.layers.Conv2D(l2filters, kernel_size=(35, 35), strides=(
        3, 3), padding='same', activation='relu')(base)

    flat = keras.layers.Flatten()(base)

    dense = keras.layers.Dense(d1width, activation='relu')(flat)
    dense = keras.layers.Dropout(do1)(dense)

    dense = keras.layers.Dense(d2width, activation='relu')(dense)
    dense = keras.layers.Dropout(do2)(dense)

    dense = keras.layers.Dense(d3width, activation='relu')(dense)
    dense = keras.layers.Dropout(do3)(dense)

    output = keras.layers.Dense(1)(dense)

    return keras.Model(inputs=input_img, outputs=output)


def model_gen_v5(input_time_dim, input_freq_dim, l1filters=64, l2filters=64, d1width=4096, d2width=512, d3width=512, do1=0.05, do2=0.005, do3=0.001):
    input_img = keras.layers.Input(shape=(input_freq_dim, input_time_dim, 1))

    base = keras.layers.Conv2D(l1filters, kernel_size=(35, 35), strides=(
        13, 13), padding='same', activation='relu')(input_img)

    base = keras.layers.Conv2D(l2filters, kernel_size=(35, 35), strides=(
        3, 3), padding='same', activation='relu')(base)

    flat = keras.layers.Flatten()(base)

    dense = keras.layers.Dense(d1width, activation='relu')(flat)
    dense = keras.layers.Dropout(do1)(dense)

    dense = keras.layers.Dense(d2width, activation='relu')(dense)
    dense = keras.layers.Dropout(do2)(dense)

    dense = keras.layers.Dense(d3width, activation='relu')(dense)
    dense = keras.layers.Dropout(do3)(dense)

    output = keras.layers.Dense(1, activation='linear')(dense)

    return keras.Model(inputs=input_img, outputs=output)
