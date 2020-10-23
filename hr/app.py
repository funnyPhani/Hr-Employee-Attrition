from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load (open ('model.pkl','rb'))
cols = ['monthlyincome', 'monthlyrate', 'dailyrate', 'totalworkingyears',
       'yearsatcompany', 'yearsincurrentrole', 'yearswithcurrmanager',
       'ageyears', 'overtime', 'distancefromhome', 'stockoptionlevel',
       'joblevel', 'maritalstatus', 'jobrole', 'yearssincelastpromotion']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = model.predict(data_unseen)
    prediction = int(prediction)
	
	
    return render_template('home.html',pred='Status of the employee is {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    #app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)
	app.run(debug=True)
