from flask import Flask, render_template, request
import pickle
import os
import pandas as pd
import numpy as np
import math
from datetime import datetime

# Load preprocessing functions
def convert_form_time_to_time_obj(time_str):
    try:
        if len(time_str) == 5:  # "HH:MM" format
            return datetime.strptime(time_str, "%H:%M").time()
        return datetime.strptime(time_str, "%H:%M:%S").time()
    except ValueError:
        return datetime.strptime("12:00", "%H:%M").time()

def haversine(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    radius = 6371.0  # Earth radius in km
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c

def load_pickle(filename):
    return pickle.load(open(os.path.join(os.pardir, 'models', filename), 'rb'))

# Load model, scaler, and encoders
model = load_pickle('rf.pkl')
scaler = load_pickle('ss.pkl')
encoders = {
    'City': load_pickle('City.pkl'),
    'Type_of_order': load_pickle('Type_of_order.pkl'),
    'Type_of_vehicle': load_pickle('Type_of_vehicle.pkl'),
    'Road_traffic_density': load_pickle('Road_traffic_density.pkl'),
    'Time_Orderd': load_pickle('Time_Orderd.pkl'),
    'Time_Order_picked': load_pickle('Time_Order_picked.pkl'),
    'Festival': load_pickle('Festival.pkl'),
    'Weatherconditions': load_pickle('Weatherconditions.pkl')
}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            form_data = request.form
            
            # Calculate distance
            rest_lat = float(form_data['Restaurant_latitude'])
            rest_lon = float(form_data['Restaurant_longitude'])
            deliv_lat = float(form_data['Delivery_location_latitude'])
            deliv_lon = float(form_data['Delivery_location_longitude'])
            distance_km = haversine(rest_lat, rest_lon, deliv_lat, deliv_lon)
            distance_km_transformed = np.sqrt(distance_km)

            # Process times
            time_orderd_obj = convert_form_time_to_time_obj(form_data['Time_Orderd'])
            time_order_picked_obj = convert_form_time_to_time_obj(form_data['Time_Order_picked'])
            time_orderd_str = time_orderd_obj.strftime('%H:%M:%S')
            time_order_picked_str = time_order_picked_obj.strftime('%H:%M:%S')

            # Helper function for encoding with NaN handling
            def encode(field, value, default=None):
                value = value.strip()
                if value.lower() in ["", "nan", "null", "none"] and default:
                    value = default
                return encoders[field].transform([value])[0]
            
            # Encode all fields
            weather_encoded = encode('Weatherconditions', form_data['Weatherconditions'])
            road_traffic_encoded = encode('Road_traffic_density', form_data['Road_traffic_density'], 'NaN')
            type_order_encoded = encode('Type_of_order', form_data['Type_of_order'])
            type_vehicle_encoded = encode('Type_of_vehicle', form_data['Type_of_vehicle'].lower())
            festival_encoded = encode('Festival', form_data['Festival'].lower(), 'NaN')
            city_encoded = encode('City', form_data['City'])
            time_orderd_encoded = encode('Time_Orderd', time_orderd_str)
            time_order_picked_encoded = encode('Time_Order_picked', time_order_picked_str)

            # Build feature array
            features = [[
                float(form_data['Delivery_person_Age']),
                np.sqrt(float(form_data['Delivery_person_Ratings'])),
                rest_lat,
                rest_lon,
                deliv_lat,
                deliv_lon,
                time_orderd_encoded,
                time_order_picked_encoded,
                weather_encoded,
                road_traffic_encoded,
                float(form_data['Vehicle_condition']),
                type_order_encoded,
                type_vehicle_encoded,
                float(form_data['multiple_deliveries']),
                festival_encoded,
                city_encoded,
                distance_km_transformed
            ]]
            
            feature_names = [
                'Delivery_person_Age', 'Delivery_person_Ratings',
                'Restaurant_latitude', 'Restaurant_longitude',
                'Delivery_location_latitude', 'Delivery_location_longitude',
                'Time_Orderd', 'Time_Order_picked',
                'Weatherconditions', 'Road_traffic_density',
                'Vehicle_condition', 'Type_of_order',
                'Type_of_vehicle', 'multiple_deliveries',
                'Festival', 'City', 'Distance_km'
            ]
            
            # Scale and predict
            features_df = pd.DataFrame(features, columns=feature_names)
            scaled_features = scaler.transform(features_df)
            prediction = model.predict(scaled_features)[0]
            
            return render_template('output.html', prediction=f"Predicted Delivery Time: {prediction:.2f} minutes")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return render_template('output.html', prediction=f"Input error: {str(e)}")
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
