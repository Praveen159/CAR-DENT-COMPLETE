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
main_folder=os.path.join(dirname(realpath(__file__)),"Images")
app.config['UPLOAD_FOLDER'] = main_folder
@app.route('/')
def input():
    return render_template("image_input.html")

@app.route('/output',methods=['GET','POST'])
def image():
    if request.method=='POST':
        f=request.files['file']
        fil=secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], fil))
        path=os.path.join(app.config['UPLOAD_FOLDER'], fil)
        output=model.predic(path)
        return render_template("image_output.html",result=output)
        
if __name__=='__main__':
    app.run(debug=False)