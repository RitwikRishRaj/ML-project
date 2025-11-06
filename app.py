from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
import os
import warnings
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store the model and test data
model = None
X_test = None
Y_test = None

# Load or train the model
def load_model():
    global model, X_train, X_test, Y_train, Y_test
    model_file = 'insurance_model.joblib'
    
    if os.path.exists(model_file):
        try:
            # Load the trained model if it exists
            model = joblib.load(model_file)
            # We need to recreate the test data for evaluation
            X_train, X_test, Y_train, Y_test = prepare_data()
            print("Loaded existing model")
            return
        except Exception as e:
            print(f"Error loading model: {e}. Retraining...")
    
    # If we get here, either the model file doesn't exist or there was an error loading it
    train_model()

def prepare_data():
    # Load the dataset
    insurance_dataset = pd.read_csv('Medicalpremium.csv')
    
    # Data preprocessing - all columns are already numeric
    # No encoding needed
    
    # Select all features except the target
    X = insurance_dataset.drop(columns='PremiumPrice', axis=1)
    Y = insurance_dataset['PremiumPrice']
    
    # Split the data with the same random_state as original
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    
    return X_train, X_test, Y_train, Y_test

def train_model():
    global model, X_test, Y_test
    
    print("Training new model...")
    # Load and preprocess data
    insurance_dataset = pd.read_csv('Medicalpremium.csv')
    X = insurance_dataset.drop(columns='PremiumPrice', axis=1)
    Y = insurance_dataset['PremiumPrice']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # Train a single, well-suited model (Random Forest)
    chosen_model_name = 'Random Forest'
    rf_model = RandomForestRegressor(
        n_estimators=200,
        random_state=2,
        max_depth=20,
        min_samples_split=5
    )
    rf_model.fit(X_train.values, Y_train.values)

    # Evaluate
    test_pred = rf_model.predict(X_test)
    r2_test = metrics.r2_score(Y_test, test_pred)
    mae_test = metrics.mean_absolute_error(Y_test, test_pred)

    print("\n" + "="*60)
    print(f"Model: {chosen_model_name}")
    print(f"Test R² score: {r2_test:.4f}")
    print(f"Mean Absolute Error: INR {mae_test:.2f}")
    print("="*60)

    # Use the trained model
    model = rf_model
    
    # Save the model
    joblib.dump(model, 'insurance_model.joblib')
    print("\nModel trained and saved successfully!")
    
    # Print a test prediction to verify
    # Age, Diabetes, BloodPressureProblems, AnyTransplants, AnyChronicDiseases, Height, Weight, KnownAllergies, HistoryOfCancerInFamily, NumberOfMajorSurgeries
    test_input = (45, 0, 0, 0, 0, 155, 57, 0, 0, 0)
    test_input_array = np.asarray(test_input)
    test_input_reshaped = test_input_array.reshape(1, -1)
    test_pred = model.predict(test_input_reshaped)
    print(f"Test prediction for (45, 0, 0, 0, 0, 155, 57, 0, 0, 0): INR {test_pred[0]:.2f}")
    print(f"Actual value from dataset: INR 25,000")
    print(f"Difference: INR {abs(test_pred[0] - 25000):.2f}")

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({'status': 'error', 'error': 'Invalid or missing JSON body'}), 400

        if model is None:
            return jsonify({'status': 'error', 'error': 'Model not loaded'}), 503
        
        # Validate input data
        required_fields = ['age', 'diabetes', 'blood_pressure_problems', 'any_transplants', 'any_chronic_diseases', 'height', 'weight', 'known_allergies', 'history_of_cancer_in_family', 'number_of_major_surgeries']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields', 'required': required_fields, 'status': 'error'}), 400
        
        # Prepare input data for prediction (matching original script's format EXACTLY)
        # Order: Age, Diabetes, BloodPressureProblems, AnyTransplants, AnyChronicDiseases, Height, Weight, KnownAllergies, HistoryOfCancerInFamily, NumberOfMajorSurgeries
        input_data = (
            int(data['age']),
            int(data['diabetes']),  # 0 for no, 1 for yes
            int(data['blood_pressure_problems']),  # 0 for no, 1 for yes
            int(data['any_transplants']),  # 0 for no, 1 for yes
            int(data['any_chronic_diseases']),  # 0 for no, 1 for yes
            float(data['height']),  # in cm
            float(data['weight']),  # in kg
            int(data['known_allergies']),  # 0 for no, 1 for yes
            int(data['history_of_cancer_in_family']),  # 0 for no, 1 for yes
            int(data['number_of_major_surgeries'])  # number of surgeries
        )
        
        # Convert to numpy array and reshape (matching original script EXACTLY)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data_reshaped)
        prediction_value = float(prediction[0])
        
        # Return prediction with same format as original script
        response = {
            'status': 'success',
            'prediction': prediction_value,
            'formatted_prediction': f'INR {prediction_value:.2f} per year',
            'period': 'annual',
            'input_data': {
                'age': int(data['age']),
                'diabetes': int(data['diabetes']),
                'blood_pressure_problems': int(data['blood_pressure_problems']),
                'any_transplants': int(data['any_transplants']),
                'any_chronic_diseases': int(data['any_chronic_diseases']),
                'height': float(data['height']),
                'weight': float(data['weight']),
                'known_allergies': int(data['known_allergies']),
                'history_of_cancer_in_family': int(data['history_of_cancer_in_family']),
                'number_of_major_surgeries': int(data['number_of_major_surgeries'])
            },
            'model_info': {
                'model_type': type(model).__name__,
                'train_r2': (metrics.r2_score(Y_train, model.predict(X_train))
                             if (model is not None and 'X_train' in globals() and X_train is not None and Y_train is not None)
                             else 'N/A'),
                'test_r2': (metrics.r2_score(Y_test, model.predict(X_test))
                            if (model is not None and 'X_test' in globals() and X_test is not None and Y_test is not None)
                            else 'N/A')
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'message': 'Error processing prediction. Please check your input data.'
        }), 500

# Train endpoint (for retraining the model if needed)
@app.route('/train', methods=['POST'])
def train():
    try:
        train_model()
        return jsonify({'status': 'success', 'message': 'Model retrained successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    try:
        # Load the model when the server starts
        load_model()
        
        # Get port from environment variable or use default
        port = int(os.environ.get('PORT', 5000))
        
        print("\n" + "="*50)
        print("Insurance Cost Prediction API")
        print("="*50)
        print(f"\nAPI is running on http://localhost:{port}")
        print("\nAvailable endpoints:")
        print(f"  - GET  http://localhost:{port}/health")
        print(f"  - POST http://localhost:{port}/predict")
        print(f"  - POST http://localhost:{port}/train")
        
        # Print example request
        print("\nExample cURL request:")
        print('''
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"age\": 48, \"gender\": 0, \"bmi\": 26.2, \"smoker\": 1, \"alcohol_consumption\": 1, \"annual_income\": 561000, \"region\": \"urban\", \"pre_existing_conditions\": 1}"
''')
        print("\nPress Ctrl+C to stop the server")
        print("="*50 + "\n")
        
        # Run the app using waitress for production
        from waitress import serve
        serve(app, host="0.0.0.0", port=port)
        
    except Exception as e:
        print(f"\nError starting the server: {str(e)}")
        print("Please make sure all dependencies are installed and the Medicalpremium.csv file exists.")
        print("You can install dependencies using: pip install -r requirements.txt")
        raise
