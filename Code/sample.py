from keras.models import load_model
import numpy as np
#from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import img_to_array,load_img
import os
os.getcwd()


Damage=load_model("C:/Users/hi/POC/Model_Fine/Damage/epoch_2.h5")
Severe=load_model("C:/Users/hi/POC/Model_Fine/Severity/sever_model_epoch2.h5")
img_256=load_img("D:/DS/Train_Images/CarDentImages/CarDentImages1.jpg",target_size=(256,256))
x=img_to_array(img_256)
x=x.reshape((1,)+x.shape)/255
predi_severe=Severe.predict(x)
pred_severe_1=np.argmax(predi_severe,axis=1)
dic_severe={0:'Minor',1:'Major',2:'Severe'}
pred_severe_1[0]
dic_severe[0]
for key in dic_severe.keys:
    if pred_severe_1[0]==key:
        print(key)
        
def predic(img_path):
	img_256=load_img(img_path,target_size=(256,256))
	x=img_to_array(img_256)
	x=x.reshape((1,)+x.shape)/255
	predi_damage=Damage.predict(x)
	if predi_damage<0.5:
		damag="Car Got Damaged"
		predi_severe=Severe.predict(x)
		pred_severe_1=np.argmax(predi_severe,axis=1)
		dic_severe={0:'Minor',1:'Major',2:'Severe'}
		severeity_pred=dic_severe[pred_severe_1]
		return damag,severeity_pred
	else:
		c="Please Submit  damaged  CAR Image"
		return c
    
output=predic("D:/DS/Train_Images/CarDentImages/CarDentImages1.jpg")