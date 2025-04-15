from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Column names and dropdown options
dropdowns = {
    'Age_band_of_driver': ['18-30', '31-50', 'under 18', 'over 51', 'unknown'],
    'Sex_of_driver': ['male', 'female', 'unknown'],
    'Educational_level': ['above high school', 'junior high school', 'unknown', 'elementary school', 'high school', 'illiterate', 'writing & reading'],
    'Vehicle_driver_relation': ['employee', 'unknown', 'owner', 'other'],
    'Driving_experience': ['1-2yr', 'above 10yr', '5-10yr', '2-5yr', 'unknown', 'no licence', 'below 1yr'],
    'Lanes_or_Medians': ['unknown', 'undivided two way', 'other', 'double carriageway (median)', 'one way', 'two-way (divided with solid lines road marking)', 'two-way (divided with broken lines road marking)'],
    'Types_of_Junction': ['no junction', 'y shape', 'crossing', 'o shape', 'other', 'unknown', 't shape', 'x shape'],
    'Road_surface_type': ['asphalt roads', 'earth roads', 'unknown', 'asphalt roads with some distress', 'gravel roads', 'other'],
    'Light_conditions': ['daylight', 'darkness - lights lit', 'darkness - no lighting', 'darkness - lights unlit'],
    'Weather_conditions': ['normal', 'raining', 'raining and windy', 'cloudy', 'other', 'windy', 'snow', 'unknown', 'fog or mist'],
    'Type_of_collision': ['collision with roadside-parked vehicles', 'vehicle with vehicle collision', 'collision with roadside objects', 'collision with animals', 'other', 'rollover', 'fall from vehicles', 'collision with pedestrians', 'with train', 'unknown'],
    'Vehicle_movement': ['going straight', 'u-turn', 'moving backward', 'turnover', 'waiting to go', 'getting off', 'reversing', 'unknown', 'parked', 'stopping', 'overtaking', 'other', 'entering a junction'],
    'Pedestrian_movement': ['not a pedestrian', "crossing from driver's nearside", 'crossing from nearside - masked by parked or stationot a pedestrianry vehicle', 'unknown or other', 'crossing from offside - masked by  parked or stationot a pedestrianry vehicle', 'in carriageway, stationot a pedestrianry - not crossing  (standing or playing)', 'walking along in carriageway, back to traffic', 'walking along in carriageway, facing traffic', 'in carriageway, stationot a pedestrianry - not crossing  (standing or playing) - masked by parked or stationot a pedestrianry vehicle'],
    'Cause_of_accident': ['moving backward', 'overtaking', 'changing lane to the left', 'changing lane to the right', 'overloading', 'other', 'no priority to vehicle', 'no priority to pedestrian', 'no distancing', 'getting off the vehicle improperly', 'improper parking', 'overspeed', 'driving carelessly', 'driving at high speed', 'driving to the left', 'unknown', 'overturning', 'turnover', 'driving under the influence of drugs', 'drunk driving']
}

@app.route('/')
def home():
    return render_template('index.html', dropdowns=dropdowns)

@app.route('/predict', methods=['POST'])
def predict():
    data = [request.form[col] for col in dropdowns.keys()]
    
    # Simulate label encoding using hash (replace with actual encoder if needed)
    encoded_data = [hash(val) % 100 for val in data]
    
    input_array = np.array(encoded_data).reshape(1, -1)

    scaler = pickle.load(open(r"C:\Accident analyser\code.ipynb\artifacts\scaler.pkl", "rb"))
    model = pickle.load(open(r"C:\Accident analyser\code.ipynb\artifacts\xgb_model.pkl", "rb"))

    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]

    return render_template('index.html', dropdowns=dropdowns, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
