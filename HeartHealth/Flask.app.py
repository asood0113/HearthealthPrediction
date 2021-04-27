from flask import Flask, render_template, request
import pickle
import numpy as np

#from keras.models import load_model

#model = load_model('RFC.pkl')
model = pickle.load(open('HH_RFC.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('hhindex.html')


@app.route('/predict', methods=['POST'])
def index():
    a = request.form['a']
    b= request.form['b']
    c = request.form['c']
    d = request.form['d']
    e=request.form['e']
    f=request.form['f']
    g=request.form['g']
    h=request.form['h']

    arr = np.array([[a,b,c,d,e,f,g,h]])
    pred = model.predict(arr)
    return render_template('model.html', data=pred)


if __name__ == "__main__":
    app.run()
