#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 14:45:57 2021

@author: akmami
"""

from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from keras.layers import Dropout
from keras.constraints import maxnorm
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import keras
import time

# load dataset
train_data = pd.read_csv('/Users/akmami/sign_mnist_train.csv')
train_data = train_data.to_numpy()
test_data = pd.read_csv('/Users/akmami/sign_mnist_test.csv')
test_data = test_data.to_numpy()
trn_ftr = train_data[:, 1:] / 255
trn_lbl = train_data[:, 0]
tst_lbl = test_data[:, 0]
labels = np.arange(10, 25)
for l in labels:
    trn_lbl[trn_lbl == l] = l - 1
    tst_lbl[tst_lbl == l] = l - 1
tst_ftr = test_data[:, 1:] / 255

activation_functions = ['relu', 'PReLU', 'sigmoid']

accuracies = []
duration_times = []
histories = []

for activation_f in activation_functions:
    print("Execution of ANN training with drop out and with " + activation_f)
    print()
    start = time.time()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    
    # create model
    model = Sequential([Dense(1024, activation=activation_f, kernel_constraint=maxnorm(3), name='Hidden_Layer_1'),
                        keras.layers.Dropout(0.2, name='Dropout_Layer_1'), 
                        Dense(512, activation=activation_f, kernel_constraint=maxnorm(3), name='Hidden_Layer_2'),
                        keras.layers.Dropout(0.2, name='Dropout_Layer_2'),
                        Dense(24, activation='softmax', name='Output_Layer')],
                       'ANN_with_dropout_and_with_' + activation_f)
    
    # Compile model
    model.compile(optimizer="adam", 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
    
    history = model.fit(trn_ftr, trn_lbl, epochs=50, validation_split=0.2, callbacks=[callback])
    prediction_ANN = model.predict(tst_ftr)
    
    end = time.time()
    
    model.summary()
    
    predict_Test_ANN = np.argmax(prediction_ANN, axis=1)
    matrix = confusion_matrix(tst_lbl, predict_Test_ANN)
    fig, ax = plot_confusion_matrix(conf_mat=matrix, figsize=(6, 6), cmap=plt.cm.Greens)
    
    histories.append(history)
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix of ANN model with dropout & with ' + activation_f, fontsize=18)
    plt.show()
    ac_test = (predict_Test_ANN == tst_lbl) * 1
    test_ac = np.sum(ac_test) / 7172
    accuracies.append((activation_f, test_ac))
    duration_times.append(end - start)
    print("End of training model with activation function = ", activation_f)
    print("--------------------------------------------------------------------------------------")
    print()
    print(classification_report(tst_lbl, predict_Test_ANN,
                                labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                        22, 23]))
    


loss_relu = histories[0].history['loss']
plt.figure()
plt.title('Training Loss Value for ANN Model with dropout & with ' + activation_functions[0])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(loss_relu)
plt.show()
loss_prelu = histories[1].history['loss']
plt.figure()
plt.title('Training Loss Value for ANN Model with dropout & with ' + activation_functions[1])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(loss_prelu)
plt.show()
loss_sigmo = histories[2].history['loss']
plt.figure()
plt.title('Training Loss Value for ANN Model with dropout & with ' + activation_functions[2])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(loss_sigmo)
plt.show()

# validation loss for relu
history = histories[0].history['val_loss']
plt.figure()
plt.title('Validation Loss Value for ANN Model with dropout & with ' + activation_functions[0])
plt.plot(history)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
# loss values for prelu
history = histories[1].history['val_loss']
plt.figure()
plt.title('Validation Loss Value for ANN Model with dropout & with ' + activation_functions[1])
plt.plot(history)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
# loss values for sigmoid
history = histories[2].history['val_loss']
plt.figure()
plt.title('Validation Loss Value for ANN Model with dropout & with ' + activation_functions[2])
plt.plot(history)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


print('Train time(secs) with dropout ' + activation_functions[0] + ' \t' + str(duration_times[0]))
print('Train time(secs) with dropout ' + activation_functions[1] + ' \t' + str(duration_times[1]))
print('Train time(secs) with dropout ' + activation_functions[2] + ' \t' + str(duration_times[2]))