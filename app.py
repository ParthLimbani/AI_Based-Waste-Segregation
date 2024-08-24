from flask import Flask, render_template, request
from keras.models import load_model
import PIL.Image as pi
import numpy as np
app = Flask(__name__)

model = load_model('waste.h5')

def predict(file):
    img = pi.open(file)
    print(img.size)
    img = img.resize([224, 224])
    print(img.size)
    img = np.array(img)
    out = model.predict(img.reshape(-1, *img.shape))
    out = np.argmax(out)
    return out

classes = {0: "Organic", 1: "Recyclable"}
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            return "No file part"
        imagefile = request.files['imagefile']
        print("+"*20, imagefile.filename)
        imagefile.save("imgdata/" + imagefile.filename)
        if imagefile.filename == '':
            return "No selected file"
        image_path = "imgdata/" + imagefile.filename  # Corrected path

        prediction = predict(image_path)
        print("*"*20, prediction)
        return render_template('index.html',prediction=classes[prediction])




    return render_template('index.html')  # Consider passing the prediction result to the template

if __name__ == '__main__':
    app.run(port=3000, debug=True)
