# 🚀 CS114 Project

A Flask-based project using PostgreSQL for database operations.

---

## 📥 Clone the Repository

```bash
git clone https://github.com/dpduy123/CS114_Project.git
cd CS114_Project
```

## Run Localhost 
### 1. 📦 Install Required Libraries
```bash
pip install flask flask_sqlalchemy psycopg2-binary


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

## 🛠️ Technologies Used
- Frontend: HTML/CSS, JavaScript
- Backend: Python, Flask, SQLAlchemy
- Database: PostgreSQL


