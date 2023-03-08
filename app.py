import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
# from sklearn.linear_model import base
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
model = pickle.load(open('life_expectancy_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = np.array(int_features).reshape(1, -1)

    # Normalize
    minmax = MinMaxScaler()
    final_features_norm = minmax.fit_transform(final_features)

    # Making Predictions
    # prediction = model.predict(final_features_norm)
    prediction = model.predict(final_features_norm)


    output = int(prediction[0][0])

    return render_template('index.html', prediction_text='Estimated Life Expectancy: {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)