# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 12:27:59 2018

@author: hi
"""

import urllib
from IPython.display import Image, display, clear_output
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline

import json
from sklearn.metrics import classification_report, confusion_matrix

sns.set_palette("cubehelix")
sns.set_style('whitegrid')

import os
import h5py
import numpy as np
import pandas as pd
import keras
keras.__version__
np.__version__
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.models import Sequential, load_model
m_damage=load_model("C:/Users/hi/POC/Model_Fine/Damage/epoch_2.h5")
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.callbacks import ModelCheckpoint, History
from keras import applications
from keras.applications.vgg16 import VGG16

location = 'C:/Users/hi/POC/Fine_Tune/Severity'
top_model_weights_path=location+'/top_model_weights.h5' # will be saved into when we create our model
# model_path = location + '/initial_data2_model.h5'
fine_tuned_model_path = location+'/model_severity.h5'

# dimensions of our images
img_width, img_height = 256, 256

train_data_dir = 'D:/DS/Images/data3a/training'
validation_data_dir = 'D:/DS/Images/data3a/validation'

train_samples = [len(os.listdir(train_data_dir+'/'+i)) for i in sorted(os.listdir(train_data_dir))]
nb_train_samples = sum(train_samples)
validation_samples = [len(os.listdir(validation_data_dir+'/'+i)) for i in sorted(os.listdir(validation_data_dir))]
nb_validation_samples = sum(validation_samples)

nb_epoch = 50

model1 = applications.VGG16(weights='imagenet', include_top=False,input_shape=(img_width,img_height,3))

### TRANSFER LEARNING ###


train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=8, 
                                            class_mode="categorical", 
                                            shuffle=False) 
    
#bottleneck_features_train = model1.predict_generator(train_generator, nb_train_samples)
#np.save(open(location+'/bottleneck_features_train5.npy', 'wb'), bottleneck_features_train)
    
    # repeat with the validation data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(validation_data_dir,
                                           target_size=(img_width, img_height),
                                           batch_size=8,
                                           class_mode="categorical",
                                           shuffle=False)
#bottleneck_features_validation = model1.predict_generator(test_generator, nb_validation_samples)
#np.save(open(location+'/bottleneck_features_validation5.npy', 'wb'), bottleneck_features_validation)

model = Sequential()
model.add(model1)
model.add(Flatten(input_shape=model1.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='sigmoid'))

for layer in model.layers[:25]:
    layer.trainable=False
 
model1.layers
model.compile(optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),loss='categorical_crossentropy',
                  metrics=['accuracy'])
checkpoint = ModelCheckpoint(fine_tuned_model_path, monitor='val_acc', 
                                 verbose=1, save_best_only=True, 
                                 save_weights_only=False, mode='auto')
    # fine-tune the model
#fit
model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              nb_epoch=2,
                              
                              validation_data=test_generator,
                              
                              verbose=1,
                              ) 

a=model.save("C:/Users/hi/POC/Model_Fine/Severity/sever_model_epoch2_bs100.h5")
model.save_weights("C:/Users/hi/POC/Fine_Tune/Severity/model1.h5")

d=load_model("C:/Users/hi/POC/Model_Fine/Severity/sever_model_epoch2_bs100.h5")


## Validation

validation_labels_severe = np.array([0] * validation_samples[0] + 
                             [1] * validation_samples[1] +
                             [2] * validation_samples[2])

predictions_severe_e2 = d.predict_generator(test_generator)

pred_labels_severe_e2 = np.argmax(predictions_severe_e2, axis=1)
classification_report(validation_labels_severe, pred_labels_severe_e2)
cm = confusion_matrix(validation_labels_severe, pred_labels_severe_e2)






