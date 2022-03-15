# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 21:35:54 2021

@author: arala
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time

from mlxtend.plotting import plot_confusion_matrix

train_data = pd.read_csv('/Users/akmami/sign_mnist_train.csv')
train_data = train_data.to_numpy()
test_data = pd.read_csv('/Users/akmami/sign_mnist_test.csv')
test_data = test_data.to_numpy()

#reshaping feature and test data as (28,28) for each sample
trn_ftr = train_data[:,1:]/255



trn_ftr = np.reshape(trn_ftr, (len(trn_ftr[:,0]),28,28))
trn_ftr2 = trn_ftr 
plt.figure()
plt.imshow(trn_ftr2[1,:,:])

tst_ftr = test_data[:,1:]/255
tst_ftr = np.reshape(tst_ftr, (len(tst_ftr[:,0]),28,28))

trn_lbl =  train_data[:,0]
tst_lbl = test_data[:,0]

labels = np.arange(10,25)

for l in labels:
    trn_lbl[trn_lbl == l] = l-1 
    tst_lbl[tst_lbl == l] = l-1



trn_ftr = tf.convert_to_tensor(trn_ftr, dtype='float32') 
tst_ftr = tf.convert_to_tensor(tst_ftr, dtype='float32')


number_f_1 = 3
kernel_size = (3,3)
activation_m1 = 'softmax'
unit1 = 1024
unit2 = 512
activation_functions = ['relu', 'PReLU', 'sigmoid']
output_units = 24
duration_times = []
histories = []
test_accuracy_CNN = []
for activation_f in activation_functions:
    print("Execution of CNN training with drop out and with " + activation_f)
    print()
    start = time.time()
    CNN = tf.keras.Sequential([keras.layers.Conv2D(filters = number_f_1, kernel_size = kernel_size, input_shape = (28,28,1), name='first_filters'),
                               keras.layers.BatchNormalization(),
                                 keras.layers.MaxPool2D(name='MaxPool1'), 
                                 keras.layers.Conv2D(filters = number_f_1, kernel_size = kernel_size, input_shape = (28,28,1), name='second_filters'), 
                                 keras.layers.BatchNormalization(),
                                 keras.layers.MaxPool2D(name='MaxPool2'), 
                                 keras.layers.Flatten(name='Flatten_of_Convs_Output'), 
                                 keras.layers.Dense(units = unit1, activation = activation_f, name = 'Hidden_Layer_1'),
                                 keras.layers.Dropout(0.2, name='Dropout_1'),
                                 keras.layers.Dense(units = unit2, activation = activation_f, name = 'Hidden_Layer_2'),
                                 keras.layers.Dropout(0.2, name='Dropout_2'),
                                 keras.layers.Dense(units = output_units, activation = activation_m1, name='Output_Layer')],
                              'CNN_with_dropout_and_with_' + activation_f)
     
    CNN.summary()
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    CNN.compile(optimizer='adam', 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()] )
    
    
    
    history = CNN.fit(trn_ftr, trn_lbl, epochs=50, validation_split=0.2, callbacks=[callback])
    
    predictions = CNN.predict(tst_ftr)
    
    
    #visualizing the first layer of the CNN to see what it is learning.
    kernel = CNN.layers[0].get_weights()[0]
    kernel = kernel[:,:,0,:]
    #normalizing it to between 0-1 
    for i in range(3):
        kernel_minimum = kernel[:,:,i].min() 
        kernel_maximum = kernel.max()
        kernel[:,:,i] = (kernel[:,:,i]- kernel_minimum) / (kernel_maximum - kernel_minimum)
        plt.figure()
        plt.title('First filter visualization of ' + activation_f)
        plt.imshow(kernel[:,:,i])
        plt.axis('off')
        plt.show()
    
    end = time.time()
    
    duration_times.append(end-start)
    histories.append(history)
    
    predict_Test = np.argmax(predictions, axis=1)
    
    matrix = confusion_matrix(tst_lbl,predict_Test)
    
    #calculation of accuracy of trained model on test set
    ac_test = (predict_Test ==tst_lbl)*1
    test_ac = np.sum(ac_test)/7172
    
    test_accuracy_CNN.append((activation_f, test_ac))
    
    fig, ax = plot_confusion_matrix(conf_mat=matrix, figsize=(6, 6), cmap=plt.cm.Greens)
    
    
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix of CNN model with dropout & with ' + activation_f, fontsize=18)
    plt.show()
    print("End of training model with activation function = ", activation_f)
    print("--------------------------------------------------------------------------------------")
    print()
    print(classification_report(tst_lbl, predict_Test, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]))




#getting the loss value for each model with different activation function
#loss values for lossrelu
loss_relu = histories[0].history['loss']

plt.figure()
plt.title('Loss Value for CNN Model with dropout & with ' + activation_functions[0])
plt.plot(loss_relu)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()



#loss values for lossprelu
loss_prelu = histories[1].history['loss']

plt.figure()
plt.title('Loss Value for CNN Model with dropout & with ' + activation_functions[1])
plt.plot(loss_prelu)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

#loss values for sigmo
loss_sigmo = histories[2].history['loss']

plt.figure()
plt.title('Loss Value for CNN Model with dropout & with ' + activation_functions[2])
plt.plot(loss_sigmo)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()



#validation loss 
val_loss_relu = histories[0].history['val_loss']

plt.figure()
plt.title('Validation Loss Value for CNN Model with dropout & with ' + activation_functions[0])
plt.plot(val_loss_relu)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

#loss values for lossprelu
val_loss_prelu = histories[1].history['val_loss']

plt.figure()
plt.title('Validation Loss Value for CNN Model with dropout & with ' + activation_functions[1])
plt.plot(val_loss_prelu[0:8])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

#loss values for sigmo
val_loss_sigmo = histories[2].history['val_loss']

plt.figure()
plt.title('Validation Loss Value for CNN Model with dropout & with ' + activation_functions[2])
plt.plot(val_loss_sigmo[0:14])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

print('Train time(secs) with dropout ' + activation_functions[0] + ' \t' + str(duration_times[0]))
print('Train time(secs) with dropout ' + activation_functions[1] + ' \t' + str(duration_times[1]))
print('Train time(secs) with dropout ' + activation_functions[2] + ' \t' + str(duration_times[2]))