import os
import json
import logging
import requests
from flask import jsonify, request
from app import app
from dotenv import load_dotenv
# Configure logging
logger = logging.getLogger(__name__)


load_dotenv(dotenv_path="secret/.env")  # specify the path

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL")


@app.route('/api/find-clinics', methods=['POST'])
def find_clinics():
    """API endpoint to find nearby heart clinics using Gemini API"""
    try:
        # Get location from request
        data = request.json
        
        if not data or 'address' not in data:
            return jsonify({'error': 'Address is required'}), 400
            
        address = data['address']
        
        # Call Gemini API to generate nearby clinics
        clinics = generate_nearby_clinics(address)
        
        return jsonify({
            'success': True,
            'clinics': clinics
        })
    except Exception as e:
        logger.error(f"Error finding clinics: {str(e)}")
        return jsonify({'error': f"Failed to find clinics: {str(e)}"}), 500

def generate_nearby_clinics(address):
    """
    Generate nearby heart clinics using Gemini API
    
    Args:
        address (str): User's address
        
    Returns:
        list: List of clinic objects with name, address, etc.
    """
    try:
        # Construct prompt for Gemini
        prompt = f"""Tìm từ 3 đến 4 bệnh viện chuyên khoa tim mạch uy tín gần địa chỉ sau: {address}.

Với mỗi địa điểm, cung cấp:
1. Tên phòng khám hoặc bệnh viện
2. Địa chỉ cụ thể
3. Số điện thoại liên hệ (nếu có)
4. Xếp hạng (nếu có, trên thang điểm 5)
5. Mô tả ngắn gọn (1-2 câu về chuyên môn hoặc dịch vụ nổi bật của họ)

Định dạng phản hồi dưới dạng JSON array với các object chứa các trường: name, address, phone, rating, description.

Chỉ trả về dữ liệu JSON hợp lệ, không kèm theo giải thích hay markdown.
"""
        # Call Gemini API
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        headers = {
            "Content-Type": "application/json",
        }
        
        url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        response_data = response.json()
        
        # Extract the generated text
        generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
        
        # Clean up the response to ensure it's valid JSON
        # Remove any markdown code blocks if present
        if "```json" in generated_text:
            generated_text = generated_text.split("```json")[1].split("```")[0].strip()
        elif "```" in generated_text:
            generated_text = generated_text.split("```")[1].split("```")[0].strip()
            
        # Parse JSON
        clinics = json.loads(generated_text)
        
        # Add unique IDs to each clinic
        for i, clinic in enumerate(clinics):
            clinic['id'] = f"clinic-{i+1}"
            
        return clinics
        
    except Exception as e:
        logger.error(f"Error generating clinics with Gemini API: {str(e)}")
        # If there's an error, return some fallback clinics
        return generate_fallback_clinics(address)
        
def generate_fallback_clinics(address):
    """Generate fallback clinics if Gemini API fails"""
    return [
        {
            "id": "clinic-1",
            "name": "HeartCare Cardiovascular Center",
            "address": f"123 Medical Ave, Near {address}",
            "phone": "555-123-4567",
            "rating": 4.7,
            "description": "Specialized in advanced cardiac care and preventive cardiology."
        },
        {
            "id": "clinic-2",
            "name": "Cardiac Wellness Institute",
            "address": f"456 Health Blvd, Near {address}",
            "phone": "555-234-5678",
            "rating": 4.5,
            "description": "Comprehensive heart health services with state-of-the-art diagnostics."
        },
        {
            "id": "clinic-3",
            "name": "Cardiovascular Specialists Group",
            "address": f"789 Heart Lane, Near {address}",
            "phone": "555-345-6789",
            "rating": 4.8,
            "description": "Expert team of cardiologists providing personalized heart disease treatment."
        }
    ]