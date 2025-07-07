from flask import Flask, render_template, request
import joblib
import pandas as pd
import traceback

app = Flask(__name__)

# ✅ Load the model
try:
    model = joblib.load('model/project_e_gradient_boost_model.pkl')
    print("✅ Model loaded successfully.")
except Exception as e:
    model = None
    print("❌ Failed to load model:", e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction="❌ Model not loaded.")

    try:
        # ✅ Collect inputs
        age = int(request.form['age'])
        gender = request.form['gender']
        screen_time = float(request.form['screen_time'])
        sleep_hours = float(request.form['sleep_hours'])
        diet = request.form['diet_quality']
        posture = request.form['posture']
        dry_eye = request.form['dry_eye']
        eye_strain = request.form['eye_strain']
        water_intake = float(request.form['water_intake'])
        family_history = request.form['family_history']
        physical_activity = request.form['physical_activity']
        outdoor_time = float(request.form['outdoor_time'])
        devices_used = request.form.getlist('devices')
        device_str = ",".join(devices_used) if devices_used else "None"

        # ✅ Create DataFrame
        input_df = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Screen_Time (hrs)': screen_time,
            'Sleep_Hours': sleep_hours,
            'Diet_Quality': diet,
            'Posture': posture,
            'Dry_Eye': dry_eye,
            'Device_Use': device_str,
            'Eye_Strain': eye_strain,
            'Water_Intake': water_intake,
            'Family_History': family_history,
            'Physical_Activity': physical_activity,
            'Outdoor_Time': outdoor_time
        }])

        print("🔍 Input DataFrame:")
        print(input_df)

        # ✅ Predict
        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)

        print(f"📈 Predicted Refractive Power: {prediction} D")

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        print("❌ Prediction error:\n", traceback.format_exc())
        return render_template('index.html', prediction=f"⚠️ Prediction Error: {str(e)}")

if __name__ == '__main__':
    print("🚀 Starting Project E Web App...")
    app.run(debug=True)
