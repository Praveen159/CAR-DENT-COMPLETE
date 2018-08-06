# -*- coding: utf-8 -*-
from keras.models import load_model
import numpy as np
#from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import img_to_array,load_img

Damage=load_model("C:/Users/hi/POC/Model_Fine/Damage/epoch_2.h5")
Severe=load_model("C:/Users/hi/POC/Model_Fine/Severity/sever_model_epoch2.h5")

def predic(img_path):
    img_256=load_img(img_path,target_size=(256,256))
    x=img_to_array(img_256)
    x=x.reshape((1,)+x.shape)/255
    print("YESSSSSSSSSSSSSSS")
    #Damage._make_predict_function()
    predi_damage=Damage.predict(x)
    print("INDIAAAAAAAAAAAAAAAAAA")
    if predi_damage<0.8:
        damag="Car Got Damaged"
        print("NOOOOOOOOOOOOOOOOOOO")
        predi_severe=Severe.predict(x)
        pred_severe_1=np.argmax(predi_severe,axis=1)
        dic_severe={0:'Minor',1:'Major',2:'Severe'}
        key=pred_severe_1[0]
        severeity_pred=dic_severe[key]
        
        return damag,severeity_pred
    else:
        c="Please Submit  damaged  CAR Image"
        return c

output=predic("D:/DS/Train_Images/CarDentImages/CarDentImages1.jpg")
    

	  
