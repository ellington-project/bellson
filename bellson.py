#!/usr/bin/env python3
import logging
import gc
import objgraph

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from lib.ellington_library import EllingtonLibrary, Track
from lib.audio import Audio
from lib.generator import LibraryIterator, TrackIterator

class NBatchLogger(keras.callbacks.Callback):
    """
    A Logger that log average performance per `display` steps.
    """
    def __init__(self, display):
        self.step = 0
        self.display = display
        self.metric_cache = {}

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for k in self.params['metrics']:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
        if self.step % self.display == 0:
            metrics_log = ''
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)
            print('step: {}/{} ::{}'.format(self.step,
                                          self.params['steps'],
                                          metrics_log))
            self.metric_cache.clear()
            gc.collect()
            # objgraph.show_most_common_types(limit=20)
            # objgraph.get_new_ids()

def model1(input_time_dim, input_freq_dim): 
    input_img = keras.layers.Input(shape=(input_freq_dim, input_time_dim, 1))

    base = keras.layers.Conv2D(24, kernel_size = (5, 5), strides=(2,2), padding='same', activation='relu')(input_img)
    base = keras.layers.BatchNormalization()(base)

    base = keras.layers.Conv2D(36, kernel_size = (5, 5), strides=(2,2), padding='same', activation='relu')(base)
    base = keras.layers.BatchNormalization()(base)

    base = keras.layers.Conv2D(48, kernel_size = (5, 5), strides=(2,2), padding='same', activation='relu')(base)
    base = keras.layers.BatchNormalization()(base)

    base = keras.layers.Conv2D(64, kernel_size = (3, 3), strides=(1,1), padding='same', activation='relu')(base)
    base = keras.layers.BatchNormalization()(base)

    base = keras.layers.Conv2D(64, kernel_size = (3, 3), strides=(1,1), padding='same', activation='relu')(base)
    base = keras.layers.BatchNormalization()(base)

    flat = keras.layers.Flatten()(base)

    dense = keras.layers.Dense(500, activation='relu')(flat) 
    # dense = keras.layers.Dropout(0.5)(dense)
    dense = keras.layers.BatchNormalization()(dense)

    dense = keras.layers.Dense(100, activation='relu')(dense) 
    # dense = keras.layers.Dropout(0.25)(dense)
    dense = keras.layers.BatchNormalization()(dense)

    dense = keras.layers.Dense(20, activation='relu')(dense) 
    # dense = keras.layers.Dropout(0.5)(dense)
    dense = keras.layers.BatchNormalization()(dense)
    
    output = keras.layers.Dense(1)(dense)

    return keras.Model(inputs = input_img, outputs = output)            

def model2(input_time_dim, input_freq_dim): 
    input_img = keras.layers.Input(shape=(input_freq_dim, input_time_dim, 1))

    base = keras.layers.Conv2D(64, kernel_size = (3, 3), strides=(1,1), padding='same', activation='relu')(input_img)
    # base = keras.layers.BatchNormalization()(base)


    base = keras.layers.Conv2D(64, kernel_size = (5, 15), strides=(4,2), padding='same', activation='relu')(base)
    # base = keras.layers.BatchNormalization()(base)

    base = keras.layers.Conv2D(36, kernel_size = (15, 5), strides=(2,2), padding='same', activation='relu')(base)
    # base = keras.layers.BatchNormalization()(base)

    # base = keras.layers.Conv2D(36, kernel_size = (15, 5), strides=(2,2), padding='same', activation='relu')(base)
    # base = keras.layers.BatchNormalization()(base)

    base = keras.layers.MaxPool2D(pool_size=(7,7))(base)

    # base = keras.layers.Conv2D(48, kernel_size = (5, 5), strides=(2,2), padding='same', activation='relu')(base)
    # base = keras.layers.BatchNormalization()(base)

    # base = keras.layers.Conv2D(64, kernel_size = (3, 3), strides=(1,1), padding='same', activation='relu')(base)
    # base = keras.layers.BatchNormalization()(base)

    # base = keras.layers.Conv2D(64, kernel_size = (3, 3), strides=(1,1), padding='same', activation='relu')(base)
    # base = keras.layers.BatchNormalization()(base)

    flat = keras.layers.Flatten()(base)

    dense = keras.layers.Dense(500, activation='relu')(flat) 
    # dense = keras.layers.Dropout(0.5)(dense)
    dense = keras.layers.BatchNormalization()(dense)

    dense = keras.layers.Dense(100, activation='relu')(dense) 
    # dense = keras.layers.Dropout(0.25)(dense)
    dense = keras.layers.BatchNormalization()(dense)

    dense = keras.layers.Dense(20, activation='relu')(dense) 
    # dense = keras.layers.Dropout(0.5)(dense)
    dense = keras.layers.BatchNormalization()(dense)
    
    output = keras.layers.Dense(1)(dense)

    return keras.Model(inputs = input_img, outputs = output) 

def model3(input_time_dim, input_freq_dim): 
    input_img = keras.layers.Input(shape=(input_freq_dim, input_time_dim, 1))

    base = keras.layers.Conv2D(64, kernel_size = (35, 35), strides=(13, 13), padding='same', activation='relu')(input_img)
    
    base = keras.layers.Conv2D(64, kernel_size = (35, 35), strides=(5, 5), padding='same', activation='relu')(base)

    # base = keras.layers.Conv2D(48, kernel_size = (5, 5), strides=(7,7), padding='same', activation='relu')(input_img)

    # base = keras.layers.Conv2D(64, kernel_size = (3, 3), strides=(3, 3), padding='same', activation='relu')(input_img)

    # base = keras.layers.Conv2D(48, kernel_size = (5, 5), strides=(2,2), padding='same', activation='relu')(base)

    # base = keras.layers.Conv2D(64, kernel_size = (3, 3), strides=(1,1), padding='same', activation='relu')(base)

    # base = keras.layers.Conv2D(64, kernel_size = (3, 3), strides=(1,1), padding='same', activation='relu')(base)

    flat = keras.layers.Flatten()(base)

    dense = keras.layers.Dense(1000, activation='relu')(flat) 
    dense = keras.layers.Dropout(0.05)(dense)

    dense = keras.layers.Dense(100, activation='relu')(dense) 
    dense = keras.layers.Dropout(0.005)(dense)

    dense = keras.layers.Dense(20, activation='relu')(dense) 
    dense = keras.layers.Dropout(0.001)(dense)
    
    output = keras.layers.Dense(1)(dense)

    return keras.Model(inputs = input_img, outputs = output) 

def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.ERROR)

    # Set up the data input etc.
    train_lib = EllingtonLibrary.from_file("data/example.el", 10)
    valid_lib = EllingtonLibrary.from_file("data/example.el", 12)

    print("Training with library of size: " + str(len(train_lib.tracks)))
    print("Validating with library of size: " + str(len(valid_lib.tracks)))

    training_gen = LibraryIterator(train_lib, samples=32, batchsize=32, start=30, end=150)
    validation_gen = LibraryIterator(valid_lib, samples=8, batchsize=8 * len(valid_lib.tracks), start=30, end=150, iterations=1)

    input_time_dim=1720
    input_freq_dim=256

    # Create the model, print info
    model = model3(input_time_dim, input_freq_dim)
    print(model.summary())

    # Compile the model
    adam = tf.train.AdamOptimizer(learning_rate=1e-04)
    sgd = keras.optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=adam,
                  loss='mse',
                  metrics=['mae', 'msle', 'mape'])

    # tfcallback = keras.callbacks.TensorBoard(log_dir='./logs',
    #                                        histogram_freq=0,
    #                                        write_grads=True,
    #                                        write_graph=False, 
    #                                        write_images=False, 
    #                                        batch_size=16)
    # bcallback = NBatchLogger(1)
    # model_file_name = 'models/bpm-model' + '-{epoch:03d}-{val_loss:.5f}.h5'
    # cplogger = keras.callbacks.ModelCheckpoint(model_file_name, monitor='val_mean_absolute_error', verbose=1, save_best_only=False)

    # Calculate how many epochs we want
    epochs = 100 * len(train_lib.tracks)
    
    # Star training! 
    for epoch in range(0, epochs): 
        batch = 0
        model.save('models/bpm-model-epoch%03d.h5' % epoch)
        model.save('models/bpm-model.h5')
        print("Epoch: %d / %d" % (epoch, epochs))
        for (train, target) in training_gen.batch(): 
            batch = batch + 1
            metrics = model.train_on_batch(x=train, y=target)

            metrics_log = ''
            for (val, k) in zip(metrics, model.metrics_names): 
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)
            print('step: {}/{} ::{}'.format(batch,
                                        training_gen.len(),
                                        metrics_log))
            gc.collect()
        
        for (train, target) in validation_gen.batch(): 
            print("Predicting batch")
            results = model.predict_on_batch(train).flatten().tolist()
            
            d = {} 
            for (e, r) in zip(target, results): 
                d.setdefault(e * 400, []).append(r * 400)

            for k, v in d.items(): 
                mean = np.mean(v)
                vss = ', '.join('%.2f' % i for i in v[0:10])
                print("{:0.4f} : {:0.4f} :: [{}] ".format(k, mean, vss))

            target_mean = np.mean(target) * 400
            target_std = np.std(target) * 400

            result_mean = np.mean(results) * 400
            result_std = np.std(results) * 400

            print("Target: %0.4f, %0.4f" % (target_mean, target_std))
            print("Result: %0.4f, %0.4f" % (result_mean, result_std))

            gc.collect()


if __name__ == '__main__':
    main()
