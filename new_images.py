# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 21:41:30 2018

@author: hi
"""

# -*- coding: utf-8 -*-
import urllib
from IPython.display import Image, display, clear_output
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline

import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

sns.set_style('whitegrid')

import os
import h5py
import numpy as np
import pandas as pd
from keras import applications
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.callbacks import ModelCheckpoint, History

import PIL
PIL.__version__


location = 'C:/Users/hi/POC/Model_Fine'
top_model_weights_path=location+'/top_model_weights_regula_epoch1.h5' # will be saved into when we create our model
# model_path = location + '/initial_data2_model.h5'
fine_tuned_model_path = location+'/ft_model2.h5'

# dimensions of our images
img_width, img_height = 256,256

train_data_dir = 'D:/DS/Images/data1a/training'
validation_data_dir = 'D:/DS/Images/data1a/validation'

train_samples = [len(os.listdir(train_data_dir+'/'+i)) for i in sorted(os.listdir(train_data_dir))]
nb_train_samples = sum(train_samples)
validation_samples = [len(os.listdir(validation_data_dir+'/'+i)) for i in sorted(os.listdir(validation_data_dir))]
nb_validation_samples = sum(validation_samples)

size_batch=50
nb_epoch = 50

model1 = applications.VGG16(weights='imagenet', include_top=False,input_shape=(img_width,img_height,3))

train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(train_data_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=8, 
                                            class_mode='binary', 
                                            shuffle=False) 
    

    # repeat with the validation data

test_generator = test_datagen.flow_from_directory(validation_data_dir,
                                           target_size=(img_width, img_height),
                                           batch_size=8,
                                           class_mode='binary',
                                           shuffle=False)

len(test_generator)
#sigmoid,25 layers , lr=1e-04,all are car dent
#Same data , but after increasing batch size from 5 to 10 and decreasing 
#steps pre epoch 12 car dent were 
#predicted correct.5 Cars were predicted correct
#Sigmoid 25 layers batch size 20 step per epoch 16 all are cars
model_damage = Sequential()
model_damage.add(model1)
model_damage.add(Flatten(input_shape=model1.output_shape[1:]))
model_damage.add(Dense(256, activation='relu'))
model_damage.add(Dropout(0.2))
model_damage.add(Dense(1, activation='sigmoid'))
 

#model1.add(model)
for layer in model_damage.layers[:25]:
    layer.trainable=False
 
model1.layers
model_damage.compile(optimizer=optimizers.SGD(lr=0.01, momentum=0.9),loss='binary_crossentropy',
                  metrics=['accuracy'])
checkpoint = ModelCheckpoint(fine_tuned_model_path, monitor='val_acc', 
                                 verbose=1, save_best_only=True, 
                                 save_weights_only=False, mode='auto')
# fine-tune the model
#fit
import time
start=time.time()
model_damage.fit_generator(train_generator,
                              steps_per_epoch=110,
                              nb_epoch=1,
                              
                              validation_data=test_generator,
                              
                              verbose=1,
                              callbacks=[checkpoint]) 
y=time.time()-start
print(y)
#a=model_damage.save("C:/Users/hi/POC/Fine_Tune/Severity/sever_model_epoch10.h5")
#bs=batchsize
#ep=epoch
#st=stepsize
#lr=learing
model.summary()
s=model_damage.save("C:/Users/hi/POC/Model_Fine/Damage/ep_5_st110_bs8_l0.01r.h5")
ss=model_damage.save("C:/Users/hi/POC/Model_Fine/Damage/ep_1_st110_bs8_l0.01r.h5")

##################
#DONT TOUCH
##################
new_model.compile(optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),loss='binary_crossentropy',
                  metrics=['accuracy'])
checkpoint = ModelCheckpoint(fine_tuned_model_path, monitor='val_acc', 
                                 verbose=1, save_best_only=True, 
                                 save_weights_only=False, mode='auto')
# fine-tune the model
fit
new_model.fit_generator(train_generator,
                              steps_per_epoch=33,
                              nb_epoch=3,
                              
                              validation_data=test_generator,
                              
                              verbose=1,
                              callbacks=[checkpoint]) 

s=new_model.save_weights("C:/Users/hi/POC/Model_Fine/model.h5")

d=load_model("C:/Users/hi/POC/Model_Fine/model.h5")
train_samples 
import keras
print(keras.__version__)
#################
#PREDICTIONS
#################

mod=load_model('C:/Users/hi/POC/Model_Fine/Damage/epoch_1_steps230.h5')

validation_labels_damage = np.array([0] * validation_samples[0] + 
                             [1] * validation_samples[1])

damage_prediction_e2 = model_damage.predict_generator(test_generator)
  
pred_labels_damage = [0 if i <0.5 else 1 for i in damage_prediction_e2]
a=classification_report(validation_labels_damage, pred_labels_damage)
cm = confusion_matrix(validation_labels_damage, pred_labels_damage)
import csv
with open('abc.csv','w') as writeFile:
    writ=csv.writer(writeFile)
    writ.writerow(a)

##################
#VALIDATION OF IMAGE
##################     

for layer in model1.layers[:25]:
    layer.trainable = False


#train_data = np.load(open('C:/Users/hi/POC/bottleneck_features_train1.npy',"rb"))
urllib.request.urlretrieve('https://www.carfax.com/media/zoo/images/rsz_frame-damage_85730e0a843d155e25e4b0f0e100bf65.jpg', 'save.jpg') # or other way to upload image
img = load_img('C:\\Users\\hi\\POC\\save.jpg', target_size=(img_width, img_height)) # this is a PIL image 
x = img_to_array(img) # this is a Numpy array with shape (3, 256, 256)
x = x.reshape((1,) + x.shape)/255
 # this is a Numpy array with shape (1, 3, 256, 256)
x.shape
pred = mod.predict(x)
preds = model_damage.predict(x)
print ("Validating that damage exists...")
print (preds)
if pred[0][0] <=.5:
    print ("Validation complete - proceed to location and severity determination")
else:
    print ("Are you sure that your car is damaged? Please submit another picture of the damage.")
    
        
        
