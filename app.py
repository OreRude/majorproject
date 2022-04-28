from flask import Flask,render_template,request
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename

# define the flask app
app=Flask(__name__,template_folder='template')

# load the model
model=load_model(r'C:\Users\Vivek singh\Desktop\Projects\Plant\models\apple2.h5')

def model_predict(img_path,model):
    test_image=image.load_img(img_path,target_size=(224,224))
    test_image=image.img_to_array(test_image)
    test_image=test_image/255
    test_image=np.expand_dims(test_image,axis=0)
    result=model.predict(test_image)
    return result


@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        # get the file from post request
        f=request.files['file']

        # save the file to uploads folder
        basepath=os.path.dirname(os.path.realpath('__file__'))
        file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result =model_predict(file_path,model)

        categories=['स्वस्थ पौधा','एकाधिक रोग','रस्ट फंगस','कैंकर रोग (Scab Disease)']

        # process your result for human
        pred_class = result.argmax()
        output=categories[pred_class]
        return output
    return None

if __name__=='__main__':
    app.run(debug=True,port=5500)