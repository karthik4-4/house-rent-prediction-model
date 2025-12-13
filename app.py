from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from House_Rent_Prediction.pipeline.prediction import Predict

app = Flask(__name__) 


@app.route('/',methods=['GET'])  
def homePage():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def index():
    try:
        city = request.form.get('city')
        area = float(request.form.get('area'))
        rooms = int(request.form.get('rooms'))
        bathrooms = int(request.form.get('bathrooms'))
        parking_spaces = int(request.form.get('parking_spaces'))
        floor = int(request.form.get('floor'))
        pets = request.form.get('pets')
        furnished = request.form.get('furnished')

        input_df = pd.DataFrame([{
            "City": city,
            "area": area,
            "rooms": rooms,
            "bathrooms": bathrooms,
            "parking Space": parking_spaces,
            "floor": floor,
            "pets": pets,
            "furnished": furnished
        }])

        preprocess = joblib.load(
            r"C:\house-rent-prediction-model\artifacts\data_transformation\preprocessor.joblib"
        )

        data_transformed = preprocess.transform(input_df)

        obj = Predict()
        prediction = obj.predict(data_transformed)

        result = round(prediction[0], 2)
        return render_template('results.html', prediction=str(result))

    except Exception as e:
        return f"Something went wrong: {e}"


if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080)
    app.run(host="0.0.0.0", port = 8080, debug=True)