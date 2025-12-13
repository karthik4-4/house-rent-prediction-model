# from flask import Flask, render_template, request
# import os 
# import numpy as np
# import pandas as pd
# import joblib
# import json
# from pathlib import Path
# from House_Rent_Prediction.pipeline.prediction import Predict

# app = Flask(__name__) 

# # Load artifacts once
# ARTIFACTS_DIR = Path("artifacts/data_transformation")
# SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
# FEATURES_PATH = ARTIFACTS_DIR / "features.json"

# # Load scaler and features if they exist early to fail fast or just load on successful start
# try:
#     scaler = joblib.load(SCALER_PATH)
#     with open(FEATURES_PATH, 'r') as f:
#         feature_columns = json.load(f)
# except Exception as e:
#     print(f"Error loading artifacts: {e}")
#     # We might want to handle this gracefully, but for now print error.
#     scaler = None
#     feature_columns = []

# @app.route('/',methods=['GET'])  
# def homePage():
#     return render_template("index.html")

# @app.route('/predict',methods=['POST'])
# def index():
#     if request.method == 'POST':
#         try:
#             if not scaler or not feature_columns:
#                 return "Model artifacts not loaded. Please run the training pipeline first."

#             # Inputs matching the schema and form
#             # Area, Rooms, Bathrooms, Parking, Floor, City, Pets, Furnished
            
#             # Numeric inputs
#             area = float(request.form.get('area', 0))
#             rooms = int(request.form.get('rooms', 0))
#             bathrooms = int(request.form.get('bathrooms', 0))
#             parking_spaces = int(request.form.get('parking_spaces', 0))
#             floor = int(request.form.get('floor', 0))
            
#             # Categorical inputs
#             city = request.form.get('city')
#             pets = request.form.get('pets')
#             furnished = request.form.get('furnished')
            
#             # Create a dictionary with all 0s for feature_columns
#             input_data = {col: 0 for col in feature_columns}
            
#             # Set numeric
#             if 'area' in input_data: input_data['area'] = area
#             if 'rooms' in input_data: input_data['rooms'] = rooms
#             if 'bathrooms' in input_data: input_data['bathrooms'] = bathrooms
#             if 'parking Space' in input_data: input_data['parking Space'] = parking_spaces
#             if 'floor' in input_data: input_data['floor'] = floor
            
#             # Handle One-Hot Encoding manually
#             # Cities: Mumbai, Bangalore, Chennai, Delhi, Hyderabad, Kolkata
#             # The columns will be like 'City_Chennai', 'City_Delhi', etc.
            
#             city_col = f"City_{city}"
#             if city_col in input_data:
#                 input_data[city_col] = 1
                
#             pets_col = f"pets_{pets}" # e.g. pets_yes
#             if pets_col in input_data:
#                 input_data[pets_col] = 1
                
#             furnished_col = f"furnished_{furnished}"
#             if furnished_col in input_data:
#                 input_data[furnished_col] = 1

#             # Convert to DataFrame
#             df = pd.DataFrame([input_data])
            
#             # Scale
#             scaled_data = scaler.transform(df)
            
#             # Predict
#             obj = Predict()
#             prediction = obj.predict(scaled_data)
            
#             result = round(prediction[0], 2)

#             return render_template('results.html', prediction = str(result))

#         except Exception as e:
#             print('The Exception message is: ',e)
#             return f'Something went wrong: {e}'

#     else:
#         return render_template('index.html')


# if __name__ == "__main__":
# 	# app.run(host="0.0.0.0", port = 8080)
#     app.run(host="0.0.0.0", port = 8080, debug=True)