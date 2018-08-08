from flask import Flask
from flask import render_template,request
#from keras.models import load_model
from os.path import dirname,realpath

from werkzeug import secure_filename
#from keras.preprocessing.image import load_img
#import numpy as np
import os

import model

app=Flask(__name__)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
main_folder=os.path.join(PROJECT_ROOT,"Images")
print(main_folder)
app.config['UPLOAD_FOLDER'] = main_folder
@app.route('/')
def input():
    return render_template("input.html")

@app.route('/output',methods=['GET','POST'])
def image():
    if request.method=='POST':
        f=request.files['file']
        fil=secure_filename(f.filename)
        f.save(os.path.join("C:/Users/hi/CarDentSeveri/Images",fil))
        
        path=os.path.join("C:/Users/hi/CarDentSeveri/Images", fil)
        output=model.predic(path)
        list_passed=["Please","Submit"]
        return render_template("output.html",result=output,result1=list_passed)
        
#if __name__=='__main__':
#    app.run(debug=False)