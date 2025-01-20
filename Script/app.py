# deploy the best model : Inference for segmentation and churn as a flas web service

from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the pre-trained best model and scaler
with open('random_forest_classifier_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

scaler = StandardScaler()

@app.route('/predict', methods=['POST'])
def predict():
    print("Received request at /predict")

    try:
        # Parse JSON input data
        data = request.get_json()
        print("Received data:", data)

        # Check if all required features are present
        required_features = ['Recency', 'Frequency', 'Monetary', 'R_Next_3Months']
        if not all(feature in data for feature in required_features):
            return jsonify({"error": f"Missing one or more required features: {', '.join(required_features)}"}), 400

        # Extract feature values and prepare for prediction
        recency = data['Recency']
        frequency = data['Frequency']
        monetary = data['Monetary']
        r_next_3months = data['R_Next_3Months']

        # Combine features into a single input array and reshape for model input
        features = np.array([recency, frequency, monetary, r_next_3months]).reshape(1, -1)
        
        # Apply scaling to match the model’s training conditions
        features = scaler.transform(features)

        # Perform prediction for segmentation and churn
        predictions = best_model.predict(features)
        
        # Assuming segmentation is the first target and churn is the second
        segment_label = predictions[0][0]
        churn = predictions[0][1]

        # Prepare and return response
        result = {
            'Segment_Label': int(segment_label),
            'Churn': int(churn)
        }
        print("Prediction result:", result)
        return jsonify(result)

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
