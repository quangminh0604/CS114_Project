import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from ml_models.model_factory import MODELS, MODELS_DIR

# Configure logging
logger = logging.getLogger(__name__)

def get_sample_data():
    """Generate a small sample dataset for model initialization"""
    # Number of samples and features
    n_samples = 30
    n_features = 14
    
    # Generate random features
    X = np.random.rand(n_samples, n_features)
    
    # Generate random target with 20% positive class
    y = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    
    return X, y

def train_model(model, X_train, y_train, X_test, y_test, model_name):
    """Train a model and calculate performance metrics"""
    try:
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        y_pred = model.predict(X_test_scaled)
        accuracy = np.mean(y_pred == y_test)
        
        # Save model and scaler
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, f"{model_name}_model.joblib")
        scaler_path = os.path.join(MODELS_DIR, f"{model_name}_scaler.joblib")
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"Trained and saved {model_name} model with accuracy: {accuracy:.4f}")
        return accuracy
    except Exception as e:
        logger.error(f"Error training {model_name}: {str(e)}")
        return 0.0

def initialize_models():
    """Initialize and train all ML models"""
    logger.info("Initializing machine learning models...")
    
    try:
        # Try to load training data from file
        try:
            train_data = pd.read_csv('train.csv')
            X = train_data.drop(columns=['TenYearCHD']).values
            y = train_data['TenYearCHD'].values
            logger.info("Loaded training data from train.csv")
        except Exception as e:
            logger.warning(f"Could not load training data: {str(e)}. Using sample data instead.")
            X, y = get_sample_data()
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train and save all models
        for name, model in MODELS.items():
            try:
                train_model(model, X_train, y_train, X_test, y_test, name)
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
        
        logger.info("Model initialization complete")
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")

def predict_with_model(model_name, data, return_probability=False):
    """Make predictions using a saved model"""
    try:
        # Load model and scaler
        model_path = os.path.join(MODELS_DIR, f"{model_name}_model.joblib")
        scaler_path = os.path.join(MODELS_DIR, f"{model_name}_scaler.joblib")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Model files for {model_name} not found")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Preprocess input data
        if isinstance(data, dict):
            features = pd.DataFrame([data])
        else:
            features = pd.DataFrame(data)
        
        # Scale features
        X_scaled = scaler.transform(features)
        
        # Make prediction - only return predictions, not probabilities as requested
        prediction = model.predict(X_scaled)
        
        # We just return prediction, ignoring probability parameter
        return prediction
    except Exception as e:
        logger.error(f"Error predicting with {model_name}: {str(e)}")
        raise

def evaluate_models_on_data(data, target):
    """Evaluate all models on provided data"""
    results = {}
    
    try:
        # Ensure data is in correct format
        if isinstance(data, str):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data)
            
        # Get target variable
        if isinstance(target, str):
            y_true = df[target].values
            X = df.drop(columns=[target]).values
        else:
            y_true = target
            X = df.values
            
        # Evaluate each model
        for model_name in MODELS.keys():
            try:
                model_path = os.path.join(MODELS_DIR, f"{model_name}_model.joblib")
                scaler_path = os.path.join(MODELS_DIR, f"{model_name}_scaler.joblib")
                
                if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                    results[model_name] = {"error": "Model not found"}
                    continue
                
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                # Scale the input data
                X_scaled = scaler.transform(X)
                
                # Predict
                y_pred = model.predict(X_scaled)
                
                # Calculate only accuracy metric as requested
                from sklearn.metrics import accuracy_score
                
                metrics = {
                    "accuracy": float(accuracy_score(y_true, y_pred))
                }
                
                results[model_name] = metrics
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                results[model_name] = {"error": str(e)}
                
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating models: {str(e)}")
        return {"error": str(e)}