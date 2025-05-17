# This file is for model definitions and custom model classes if needed
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class FraminghamRiskClassifier(BaseEstimator, ClassifierMixin):
    """
    Custom classifier implementing the Framingham Heart Study risk algorithm.
    This is a simplified version of the Framingham risk score algorithm.
    """
    
    def __init__(self):
        self.coefficients = {
            'male': 0.5,
            'age': 0.05,
            'cigsPerDay': 0.02,
            'BPMeds': 0.3,
            'prevalentStroke': 0.7,
            'prevalentHyp': 0.4,
            'diabetes': 0.6,
            'totChol': 0.003,
            'sysBP': 0.005,
            'diaBP': 0.003,
            'BMI': 0.02,
            'heartRate': 0.002,
            'glucose': 0.002
        }
        self.threshold = 0.5
        
    def fit(self, X, y):
        """No training needed for this rule-based model"""
        return self
        
    def predict_proba(self, X):
        """Calculate probability of heart disease risk"""
        # Calculate risk score
        risk_scores = np.zeros(X.shape[0])
        
        # Add contribution from each feature
        for i, feature in enumerate(self.coefficients.keys()):
            # Find the column index for this feature in X
            try:
                col_idx = list(X.columns).index(feature) if hasattr(X, 'columns') else i
                risk_scores += X[:, col_idx] * self.coefficients[feature]
            except (IndexError, AttributeError):
                # Skip if feature not found
                continue
        
        # Convert to probability using sigmoid function
        probabilities = 1 / (1 + np.exp(-risk_scores))
        
        # Return probabilities in the format [P(class=0), P(class=1)]
        return np.vstack([1 - probabilities, probabilities]).T
        
    def predict(self, X):
        """Predict heart disease risk (binary classification)"""
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= self.threshold).astype(int)
