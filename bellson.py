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

def build_model(input_time_dim, input_freq_dim): 
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

def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.ERROR)
    el = EllingtonLibrary.from_file("data/example.el")

    print("Training with library of size: " + str(len(el.tracks)))

    input_time_dim=860
    input_freq_dim=256

    model = build_model(input_time_dim, input_freq_dim)

    print(model.summary())

    adam = tf.train.AdamOptimizer(learning_rate=1e-04)
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=adam,
                  loss='mse',
                  metrics=['mae', 'msle', 'mape'])

    training_gen = LibraryIterator(el, samples=8, batchsize=8)
    validation_gen = LibraryIterator(el, samples=2, batchsize=2)

    tfcallback = keras.callbacks.TensorBoard(log_dir='./logs',
                                           histogram_freq=0,
                                           write_grads=True,
                                           write_graph=False, 
                                           write_images=False, 
                                           batch_size=16)
    bcallback = NBatchLogger(1)
    model_file_name = 'models/bpm-model' + '-{epoch:03d}-{val_loss:.5f}.h5'
    

    cplogger = keras.callbacks.ModelCheckpoint(model_file_name, monitor='val_mean_absolute_error', verbose=1, save_best_only=False)

    epochs = 50

    train_generator = False
    if train_generator: 
        model.fit_generator(training_gen.batch(),
                            steps_per_epoch=training_gen.len(),
                            epochs=epochs,
                            verbose=2,
                            callbacks=[tfcallback, bcallback, cplogger],
                            validation_data=validation_gen.batch(),
                            validation_steps=validation_gen.len(),
                            class_weight=None,
                            max_queue_size=60,
                            workers=1,
                            use_multiprocessing=True,
                            shuffle=True,
                            initial_epoch=0)
    else:
        for epoch in range(0, epochs): 
            batch = 0
            model.save('models/bpm-model-epoch%03d.h5' % epoch)
            for (train, target) in training_gen.batch(): 
                batch = batch + 1
                print("Training batch " + str(batch))
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
                
                print("Predicting batch")
                results = model.predict_on_batch(train).flatten().tolist()
                

                results_log = ''
                d = {} 
                for (e, r) in zip(target, results): 
                    d.setdefault(e * 400, []).append(r * 400)

                for k, v in d.items(): 
                    vss = ','.join('%.4f' % i for i in v)
                    print("{} : [{}] ".format(k, vss))

                # for (r, e) in zip(results, target): 
                #     results_log += '(%.4f, %.4f) ' % (e * 400, r * 400)
                # print('Targets - Predictions: [{}]'.format(results_log) )

                gc.collect()


if __name__ == '__main__':
    main()
