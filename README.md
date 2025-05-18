# 🚀 CS114 Project

A Flask-based machine learning web application that uses Logistic Regression, Support Vector Machine (SVM), Decision Trees, and Random Forest to predict the likelihood of heart attack based on the **Framingham Heart Study dataset**. The project also compares the performance and accuracy of these models.
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

## Run Locally
### 1. 📦 Install Required Libraries
```bash
pip install flask flask_sqlalchemy psycopg2-binary
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
export DATABASE_URI = "postgresql://username:yourpassword@localhost/mydatabase"
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

## 🧠 Machine Learning Models Used

- 🔹 Logistic Regression

- 🔹 Support Vector Machine (SVM)

- 🔹 Decision Tree

- 🔹 Random Forest

Each model is trained using the Framingham Heart Study dataset, and their prediction performance is compared using accuracy and other metrics.
## 🛠️ Technologies Used
- Frontend: HTML/CSS, JavaScript
- Backend: Python, Flask, SQLAlchemy
- Database: PostgreSQL


