import os
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from ml_models.custom_models import LogisticRegressionScratch, DecisionTreeScratch, RandomForestScratch, SVMScratch

# Configure logging
logger = logging.getLogger(__name__)

# Define models directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Feature names for Framingham Heart Disease dataset
FEATURE_NAMES = [
    'male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 
    'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 
    'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
]

# Define all models
MODELS = {
    # Sklearn models
    'logistic_regression_sklearn': LogisticRegression(random_state=42, max_iter=1000),
    'svm_sklearn': SVC(kernel='rbf', probability=True, random_state=42),
    'decision_tree_sklearn': DecisionTreeClassifier(random_state=42, max_depth=5),
    'random_forest_sklearn': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5),
    
    # From scratch models
    'logistic_regression_scratch': LogisticRegressionScratch(learning_rate=0.1, n_iters=1000),
    'svm_scratch': SVMScratch(C=1.0, gamma=0.5),
    'decision_tree_scratch': DecisionTreeScratch(max_depth=5, random_state=42),
    'random_forest_scratch': RandomForestScratch(n_trees=20, max_depth=5, random_state=42)  # Using 20 trees for faster performance
}

def load_train_data():
    """Load training data from CSV file"""
    try:
        # Check if train.csv exists in the current directory
        train_path = os.path.join(os.getcwd(), 'train.csv')
        if os.path.exists(train_path):
            df = pd.read_csv(train_path)
            # Convert target column TenYearCHD to numpy array
            X = df.drop(columns=['TenYearCHD']).values
            y = df['TenYearCHD'].values
            return X, y
        else:
            # If train.csv doesn't exist, generate sample data
            logger.warning("Train.csv not found, using sample data")
            return generate_sample_data()
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        return generate_sample_data()

def generate_sample_data():
    """Generate sample data for model training"""
    n_samples = 1000
    n_features = len(FEATURE_NAMES)
    
    # Generate random features
    X = np.random.rand(n_samples, n_features)
    
    # Generate binary target with 20% positive cases
    y = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    
    return X, y

def train_model(model, X_train, y_train, X_test, y_test, model_name):
    """Train a model and calculate performance metrics"""
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division='warn'),
        'recall': recall_score(y_test, y_pred, zero_division='warn'),
        'f1_score': f1_score(y_test, y_pred, zero_division='warn'),
    }
    
    if y_pred_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
    
    # Save model and scaler
    model_path = os.path.join(MODELS_DIR, f"{model_name}_model.joblib")
    scaler_path = os.path.join(MODELS_DIR, f"{model_name}_scaler.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    logger.info(f"Trained and saved {model_name} model with accuracy: {metrics['accuracy']:.4f}")
    
    return metrics

def initialize_models():
    """Initialize and train all ML models"""
    logger.info("Initializing machine learning models...")
    
    # Get training data
    X, y = load_train_data()
    
    # Split data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train all models
    results = {}
    for name, model in MODELS.items():
        try:
            metrics = train_model(model, X_train, y_train, X_test, y_test, name)
            results[name] = metrics
        except Exception as e:
            logger.error(f"Error training {name} model: {str(e)}")
    
    logger.info("Model initialization complete")
    return results

def predict_with_model(model_name, data, return_probability=False):
    """Make predictions using a saved model"""
    # Load model and scaler
    model_path = os.path.join(MODELS_DIR, f"{model_name}_model.joblib")
    scaler_path = os.path.join(MODELS_DIR, f"{model_name}_scaler.joblib")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        logger.error(f"Model or scaler file for {model_name} not found")
        raise FileNotFoundError(f"Model files for {model_name} not found")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Preprocess input data
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)
    
    # Ensure all required features are present
    missing_features = set(FEATURE_NAMES) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    # Scale features
    X_scaled = scaler.transform(df[FEATURE_NAMES])
    
    # Make prediction
    prediction = model.predict(X_scaled)
    
    if return_probability and hasattr(model, "predict_proba"):
        probability = model.predict_proba(X_scaled)[:, 1]
        return prediction, probability
    
    return prediction

def evaluate_models_on_data(data, target):
    """Evaluate all models on provided data"""
    # Convert data to DataFrame if needed
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    if not isinstance(target, pd.Series):
        target = pd.Series(target)
    
    results = {}
    
    for model_name in MODELS.keys():
        try:
            # Load model and scaler
            model_path = os.path.join(MODELS_DIR, f"{model_name}_model.joblib")
            scaler_path = os.path.join(MODELS_DIR, f"{model_name}_scaler.joblib")
            
            if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
                results[model_name] = {"error": f"Model files for {model_name} not found"}
                continue
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            # Ensure all required features are present
            missing_features = set(FEATURE_NAMES) - set(data.columns)
            if missing_features:
                results[model_name] = {"error": f"Missing features: {missing_features}"}
                continue
            
            # Scale features
            X_scaled = scaler.transform(data[FEATURE_NAMES])
            
            # Make predictions
            y_pred = model.predict(X_scaled)
            y_pred_proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Calculate metrics
            metrics = {
                'accuracy': float(accuracy_score(target, y_pred)),
                'precision': float(precision_score(target, y_pred, zero_division='warn')),
                'recall': float(recall_score(target, y_pred, zero_division='warn')),
                'f1_score': float(f1_score(target, y_pred, zero_division='warn')),
            }
            
            if y_pred_proba is not None:
                metrics['auc_roc'] = float(roc_auc_score(target, y_pred_proba))
                
            results[model_name] = metrics
        
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            results[model_name] = {"error": str(e)}
    
    return results