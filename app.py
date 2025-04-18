from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load('heart.joblib')

@app.route('/', methods=['GET'])
def home():
    # On initial load or reset, no result
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values from form
        data = {
            'age': [int(request.form['age'])],
            'sex': [int(request.form['sex'])],
            'cp': [int(request.form['cp'])],
            'trestbps': [int(request.form['tresbps'])],
            'chol': [int(request.form['chol'])],
            'fbs': [int(request.form['fbs'])],
            'restecg': [int(request.form['restecg'])],
            'thalach': [int(request.form['thalach'])],
            'exang': [int(request.form['exang'])],
            'oldpeak': [float(request.form['oldpeak'])],
            'slope': [int(request.form['slope'])],
            'ca': [int(request.form['ca'])],
            'thal': [int(request.form['thal'])]
        }

        input_df = pd.DataFrame(data)
        prediction = model.predict(input_df)[0]
        result = "Yes (Heart Disease)" if prediction == 1 else "No (Healthy Heart)"
        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
