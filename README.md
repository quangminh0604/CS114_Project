# 🚀 CS114 Project

A Flask-based machine learning web application that uses Logistic Regression, Support Vector Machine (SVM), Decision Trees, and Random Forest to predict the likelihood of heart attack based on the **Framingham Heart Study dataset**. The project also compares the performance and accuracy of these models.
---
## 🧠 Machine Learning Models Used

- 🔹 Logistic Regression

- 🔹 Support Vector Machine (SVM)

- 🔹 Decision Tree

- 🔹 Random Forest

Each model is trained using the Framingham Heart Study dataset, and their prediction performance is compared using accuracy and other metrics.
---
## 📊 Dataset
- 📁 Dataset: framingham.csv
- 📌 Source: Framingham Heart Study
- 🩺 Purpose: Predict the probability of cardiovascular disease occurrence using patient data such as age, cholesterol, glucose, smoking habits, and more.
---

## 📥 Clone the Repository

```bash
git clone https://github.com/dpduy123/CS114_Project.git
cd DemoApp
```
## 🛡️ Setup Gemini API Key
To use the Gemini API for finding nearby clinics:
### 1. 🔐 Get Your API Key
- Visit Google AI Studio or Google Cloud Console
- Enable the Gemini API / Generative Language API
- Create a new API key
### 2. 🗂️ Create a .env File
Create a folder named secret/ in your project root and add a file named .env inside it:
```bash
CS114_Project/DemoApp
├── app.py
├── ...
└── secret/
    └── .env
```
Add the following content to secret/.env:
```.env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_API_URL=https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent
```
⚠️ Do not share or commit this file.

### 3. ✅ Already Configured in Code
```python 
from dotenv import load_dotenv
load_dotenv(dotenv_path="secret/.env")
```

## 🖥️ Run Locally
### 1. 📦 Install Required Libraries
```bash
pip install flask flask_sqlalchemy psycopg2-binary scikit-learn pandas matplotlib seaborn python-dotenv
```

### 2. 🗃️ Create a PostgreSQL Database:
if you haven't created a database yet:
```bash
createdb -U postgres mydatabase
```

### 3. ⚙️ Configure the Database URI
🔹 Windows (PowerShell)
```powershell
$env:SQLALCHEMY_DATABASE_URI = "postgresql://username:yourpassword@localhost:5432/mydatabase"
```

🔹 MacOS/Linux
```bash
export DATABASE_URL = "postgresql://username:yourpassword@localhost/mydatabase"
```
⚠️ Replace username and yourpassword with your actual PostgreSQL credentials.


### 4. 🚀 Run the Flask App
```
python main.py
```
Your app should now be running on http://localhost:5000. (or port 5432)

## 📌 Notes

- Ensure **PostgreSQL** is installed and running.
- Make sure **environment variables** are properly set before starting the app.
- Customize your **database connection string** (`SQLALCHEMY_DATABASE_URI`) as needed.

## 🛠️ Technologies Used
- Frontend: HTML/CSS, JavaScript
- Backend: Python, Flask, SQLAlchemy
- Database: PostgreSQL


