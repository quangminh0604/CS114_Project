from app import db
from datetime import datetime

class Prediction(db.Model):
    """Model for storing prediction results"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    algorithm = db.Column(db.String(50), nullable=False)
    
    # Framingham Heart Study parameters
    male = db.Column(db.Boolean, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    currentSmoker = db.Column(db.Boolean)
    cigsPerDay = db.Column(db.Float)
    BPMeds = db.Column(db.Boolean)
    prevalentStroke = db.Column(db.Boolean)
    prevalentHyp = db.Column(db.Boolean)
    diabetes = db.Column(db.Boolean)
    totChol = db.Column(db.Float)
    sysBP = db.Column(db.Float)
    diaBP = db.Column(db.Float)
    BMI = db.Column(db.Float)
    heartRate = db.Column(db.Float)
    glucose = db.Column(db.Float)
    
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
            'male': self.male,
            'age': self.age,
            'currentSmoker': self.currentSmoker,
            'cigsPerDay': self.cigsPerDay,
            'BPMeds': self.BPMeds,
            'prevalentStroke': self.prevalentStroke,
            'prevalentHyp': self.prevalentHyp,
            'diabetes': self.diabetes,
            'totChol': self.totChol,
            'sysBP': self.sysBP,
            'diaBP': self.diaBP,
            'BMI': self.BMI,
            'heartRate': self.heartRate,
            'glucose': self.glucose,
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
