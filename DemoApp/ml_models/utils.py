import io
import pandas as pd
import csv
import logging
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

def validate_input_data(data):
    """
    Validate input data for Framingham risk prediction
    
    Args:
        data (dict): Dictionary containing patient data
        
    Returns:
        tuple: (is_valid, error_message)
    """
    required_fields = [
        'gender', 'age' , 'hypertension', 'smoking_status', 
        'bmi', 'heart_disease', 'avg_glucose_level'
    ]
    
    # Check for missing fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    # Validate data types
    try:
        # Convert boolean fields
        for field in ['hypertension', 'heart_disease']:
            if isinstance(data[field], str):
                if data[field].lower() in ['true', 'yes', '1']:
                    data[field] = True
                elif data[field].lower() in ['false', 'no', '0']:
                    data[field] = False
            data[field] = bool(data[field])
        
        # Convert numeric fields
        for field in ['age',  'bmi', 'avg_glucose_level']:
            data[field] = float(data[field])
        
        # Validate age range
        if data['age'] < 18 or data['age'] > 120:
            return False, "Age should be between 18 and 100 years"
        
        # Validate other ranges
        if data['bmi'] < 15 or data['bmi'] > 60:
            return False, "BMI should be between 15 and 60"
        
    
        if data['avg_glucose_level'] < 40 or data['avg_glucose_level'] > 400:
            return False, "Glucose level should be between 40 and 400 mg/dL"
        
        return True, "Data validation successful"
        
    except Exception as e:
        logger.error(f"Data validation error: {str(e)}")
        return False, f"Data validation error: {str(e)}"

def parse_csv_file(file_content):
    """
    Parse CSV file content to extract patient data
    
    Args:
        file_content (bytes): CSV file content
        
    Returns:
        tuple: (success, dataframe_or_error_message)
    """
    try:
        # Read CSV file
        df = pd.read_csv(io.BytesIO(file_content))
        
        # Check if required columns are present
        required_columns = [
        'gender', 'age' , 'hypertension', 'smoking_status', 
        'bmi', 'heart_disease', 'avg_glucose_level'
    ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        return True, df
        
    except Exception as e:
        logger.error(f"CSV parsing error: {str(e)}")
        return False, f"Error parsing CSV file: {str(e)}"

def generate_csv_response(data, predictions):
    """
    Generate CSV response with prediction results
    
    Args:
        data (pd.DataFrame): Input data
        predictions (np.array): Model predictions
        
    Returns:
        str: CSV content as string
    """
    try:
        # Create a copy of the input data
        result_df = data.copy()
        
        # Add prediction column
        result_df['prediction'] = predictions
        
        # Add risk level column
        result_df['risk_level'] = result_df['prediction'].apply(
            lambda p: 'High Risk' if p else 'Low Risk'
        )
        
        # Convert DataFrame to CSV string
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        
        return csv_buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error generating CSV response: {str(e)}")
        return "Error,generating,CSV,response"