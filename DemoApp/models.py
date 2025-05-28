from app import db
from datetime import datetime

class Prediction(db.Model):
    """Model for storing prediction results"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    algorithm = db.Column(db.String(50), nullable=False)
    
    # Framingham Heart Study parameters
    gender = db.Column(db.String, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    smokingStatus = db.Column(db.String, nullable=False)
    hypertension = db.Column(db.Boolean, nullable=False)
    heartDisease = db.Column(db.Boolean, nullable=False)
    avgGlucose = db.Column(db.Float, nullable=False)
    BMI = db.Column(db.Float, nullable=False)
    
    # Prediction result
    risk_prediction = db.Column(db.Boolean, nullable=False)
    
    def __repr__(self):
        return f"<Prediction {self.id}: {self.risk_prediction}>"
    
    @property
    def serialize(self):
        """Return object data in easily serializable format"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'algorithm': self.algorithm,
            'gender': self.gender,
            'age': self.age,
            'smokingStatus': self.smokingStatus,
            'hypertension': self.hypertension,
            'heartDisease': self.heartDisease,
            'avgGlucose': self.avgGlucose,
            'BMI': self.BMI,
            'risk_prediction': self.risk_prediction
        }

class ModelPerformance(db.Model):
    """Model for storing ML model performance metrics"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    model_name = db.Column(db.String(50), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    
    def __repr__(self):
        return f"<ModelPerformance {self.model_name}: {self.accuracy}>"
