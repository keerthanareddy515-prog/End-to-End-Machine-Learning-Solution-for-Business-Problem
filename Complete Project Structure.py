Complete Project Structure
churn_ml_system/
│
├── data/
│   └── churn_data.csv
│
├── model/
│   └── model.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── app/
│   ├── app.py
│   └── templates/
│       └── index.html
│
├── requirements.txt
└── README.md
📊 1. Dataset (data/churn_data.csv)

👉 Create this CSV file manually:

gender,tenure,MonthlyCharges,Contract,InternetService,Churn
Female,1,29.85,Month-to-month,DSL,1
Male,34,56.95,One year,Fiber optic,0
Male,2,53.85,Month-to-month,DSL,1
Female,45,42.30,Two year,Fiber optic,0
Female,5,70.70,Month-to-month,Fiber optic,1
Male,10,99.65,One year,DSL,0
Female,60,89.10,Two year,Fiber optic,0
Male,3,29.75,Month-to-month,DSL,1
⚙️ 2. Preprocessing (src/preprocess.py)
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df = df.copy()
    df.dropna(inplace=True)

    label_encoders = {}

    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return X, y, label_encoders
🤖 3. Training (src/train.py)
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_data, preprocess

# Load data
df = load_data("../data/churn_data.csv")

# Preprocess
X, y, encoders = preprocess(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save model
with open("../model/model.pkl", "wb") as f:
    pickle.dump((model, encoders), f)

print("Model trained and saved!")
📈 4. Evaluation (src/evaluate.py)
import pickle
from sklearn.metrics import accuracy_score, classification_report
from preprocess import load_data, preprocess

df = load_data("../data/churn_data.csv")
X, y, _ = preprocess(df)

with open("../model/model.pkl", "rb") as f:
    model, _ = pickle.load(f)

predictions = model.predict(X)

print("Accuracy:", accuracy_score(y, predictions))
print("\nReport:\n", classification_report(y, predictions))
🔮 5. Prediction (src/predict.py)
import pickle
import numpy as np

def make_prediction(input_data):
    with open("../model/model.pkl", "rb") as f:
        model, encoders = pickle.load(f)

    # Convert categorical manually (example mapping)
    input_data[0] = encoders['gender'].transform([input_data[0]])[0]
    input_data[3] = encoders['Contract'].transform([input_data[3]])[0]
    input_data[4] = encoders['InternetService'].transform([input_data[4]])[0]

    prediction = model.predict([input_data])
    return prediction[0]
🌐 6. Web App (app/app.py)
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model, encoders = pickle.load(open("../model/model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    gender = request.form["gender"]
    tenure = float(request.form["tenure"])
    monthly = float(request.form["monthly"])
    contract = request.form["contract"]
    internet = request.form["internet"]

    # Encode inputs
    gender = encoders['gender'].transform([gender])[0]
    contract = encoders['Contract'].transform([contract])[0]
    internet = encoders['InternetService'].transform([internet])[0]

    data = [[gender, tenure, monthly, contract, internet]]
    prediction = model.predict(data)[0]

    result = "Customer will Churn" if prediction == 1 else "Customer will Stay"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
🖥️ 7. HTML UI (app/templates/index.html)
<!DOCTYPE html>
<html>
<head>
    <title>Churn Prediction</title>
</head>
<body>

<h2>Customer Churn Prediction</h2>

<form action="/predict" method="post">
    Gender:
    <select name="gender">
        <option>Male</option>
        <option>Female</option>
    </select><br><br>

    Tenure:
    <input type="number" name="tenure" required><br><br>

    Monthly Charges:
    <input type="number" step="0.01" name="monthly" required><br><br>

    Contract:
    <select name="contract">
        <option>Month-to-month</option>
        <option>One year</option>
        <option>Two year</option>
    </select><br><br>

    Internet Service:
    <select name="internet">
        <option>DSL</option>
        <option>Fiber optic</option>
    </select><br><br>

    <button type="submit">Predict</button>
</form>

<h3>{{ prediction_text }}</h3>

</body>
</html>