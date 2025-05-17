import os
import io
import pandas as pd
import numpy as np
import csv
import logging
from flask import render_template, request, jsonify, redirect, url_for, flash, send_file, Response
from app import app, db
from models import Prediction, ModelPerformance
from ml_models.model_factory import predict_with_model, evaluate_models_on_data
from ml_models.utils import validate_input_data, parse_csv_file, generate_csv_response

# Setup logging
logger = logging.getLogger(__name__)

# Google Maps API key
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

# Available ML algorithms
ML_ALGORITHMS = {
    'logistic_regression_sklearn': 'Logistic Regression (sklearn)',
    'logistic_regression_scratch': 'Logistic Regression (from scratch)',
    'svm_sklearn': 'SVM (sklearn)',
    'svm_scratch': 'SVM (from scratch)',
    'decision_tree_sklearn': 'Decision Tree (sklearn)',
    'decision_tree_scratch': 'Decision Tree (from scratch)',
    'random_forest_sklearn': 'Random Forest (sklearn)',
    'random_forest_scratch': 'Random Forest (from scratch)'
}

@app.route('/')
def index():
    """Home page route"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page route"""
    return render_template('about.html')

@app.route('/symptoms')
def symptoms():
    """Symptoms page route"""
    return render_template('symptoms.html')

@app.route('/predict')
def predict():
    """Prediction page route"""
    return render_template(
        'predict.html', 
        algorithms=ML_ALGORITHMS,
        google_maps_api_key=GOOGLE_MAPS_API_KEY
    )

@app.route('/compare')
def compare():
    """Model comparison page route"""
    return render_template(
        'compare.html',
        algorithms=ML_ALGORITHMS
    )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for individual prediction"""
    try:
        # Get JSON data from request
        data = request.json
        
        # Get selected algorithm
        algorithm = data.pop('algorithm', 'logistic_regression_sklearn')
        if algorithm not in ML_ALGORITHMS:
            return jsonify({'error': f"Unknown algorithm: {algorithm}"}), 400
        
        # Validate input data
        is_valid, message = validate_input_data(data)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Automatically set currentSmoker based on cigsPerDay
        if 'cigsPerDay' in data:
            data['currentSmoker'] = data['cigsPerDay'] > 0
        elif 'currentSmoker' not in data:
            data['currentSmoker'] = False
            
        # Make prediction
        prediction = predict_with_model(algorithm, data, return_probability=False)
        
        try:
            # Store prediction in database
            new_prediction = Prediction(
                algorithm=algorithm,
                male=data['male'],
                age=data['age'],
                cigsPerDay=data['cigsPerDay'],
                BPMeds=data['BPMeds'],
                prevalentStroke=data['prevalentStroke'],
                prevalentHyp=data['prevalentHyp'],
                diabetes=data['diabetes'],
                totChol=data['totChol'],
                sysBP=data['sysBP'],
                diaBP=data['diaBP'],
                BMI=data['BMI'],
                heartRate=data['heartRate'],
                glucose=data['glucose'],
                risk_prediction=bool(prediction[0])
            )
            # Set currentSmoker separately to handle potential database schema issues
            new_prediction.currentSmoker = data['currentSmoker']
        except Exception as e:
            logger.error(f"Error creating prediction record: {str(e)}")
            # Create prediction without currentSmoker if there's a schema issue
            new_prediction = Prediction(
                algorithm=algorithm,
                male=data['male'],
                age=data['age'],
                cigsPerDay=data['cigsPerDay'],
                BPMeds=data['BPMeds'],
                prevalentStroke=data['prevalentStroke'],
                prevalentHyp=data['prevalentHyp'],
                diabetes=data['diabetes'],
                totChol=data['totChol'],
                sysBP=data['sysBP'],
                diaBP=data['diaBP'],
                BMI=data['BMI'],
                heartRate=data['heartRate'],
                glucose=data['glucose'],
                risk_prediction=bool(prediction[0])
            )
        db.session.add(new_prediction)
        db.session.commit()
        
        # Return prediction result
        return jsonify({
            'prediction': bool(prediction[0]),
            'risk_level': 'High Risk' if prediction[0] else 'Low Risk'
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def api_batch_predict():
    """API endpoint for batch prediction via CSV upload"""
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        # Get selected algorithm
        algorithm = request.form.get('algorithm', 'logistic_regression_sklearn')
        if algorithm not in ML_ALGORITHMS:
            return jsonify({'error': f"Unknown algorithm: {algorithm}"}), 400
        
        # Get file from request
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be a CSV file'}), 400
        
        # Parse CSV file
        success, result = parse_csv_file(file.read())
        if not success:
            return jsonify({'error': result}), 400
        
        # Make batch prediction
        df = result
        predictions = predict_with_model(algorithm, df, return_probability=False)
        
        # Generate CSV response
        csv_data = generate_csv_response(df, predictions)
        
        # Create a response with CSV content
        response = Response(
            csv_data,
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename=prediction_results.csv'}
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare-models', methods=['POST'])
def api_compare_models():
    """API endpoint for model comparison"""
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        # Get file from request
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be a CSV file'}), 400
        
        # Parse CSV file
        success, result = parse_csv_file(file.read())
        if not success:
            return jsonify({'error': result}), 400
        
        # Get target column from form
        target_column = request.form.get('target', 'TenYearCHD')
        
        # Check if target column exists
        if target_column not in result.columns:
            return jsonify({'error': f'CSV file must include a "{target_column}" column with actual outcomes'}), 400
        
        # Extract features and target
        df = result
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Evaluate all models
        evaluation_results = evaluate_models_on_data(X, y)
        
        # Store evaluation results
        for model_name, metrics in evaluation_results.items():
            if 'error' not in metrics:
                performance = ModelPerformance(
                    model_name=model_name,
                    accuracy=metrics.get('accuracy', 0)
                )
                db.session.add(performance)
        
        db.session.commit()
        
        # Return evaluation results
        return jsonify(evaluation_results)
    
    except Exception as e:
        logger.error(f"Model comparison error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_risk_level(probability):
    """
    Determine risk level based on probability
    
    Args:
        probability (float): Risk probability
        
    Returns:
        str: Risk level (Low, Moderate, High)
    """
    if probability < 0.1:
        return "Low"
    elif probability < 0.2:
        return "Moderate"
    else:
        return "High"
