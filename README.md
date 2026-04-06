# HR-Analytics-Portal

# 🧠 Employee Churn Prediction with Explainable AI (SHAP)

## 📌 Project Overview

This project predicts whether an employee is likely to **stay or leave** the company using Machine Learning.
It also explains the prediction using **SHAP (Explainable AI)** to show how each feature contributes to the final decision.

---

## 🚀 Features

* Predict employee churn using **XGBoost Classifier**
* Interactive **Flask Web Application**
* Explain predictions using **SHAP values**
* Visual feature contribution graph (Red = Leave, Green = Stay)
* Displays:

  * Prediction result
  * Feature importance
  * Base value & final output

---

## 🛠️ Tech Stack

* Python
* Flask
* XGBoost
* SHAP (Explainable AI)
* Scikit-learn
* Pandas, NumPy
* Matplotlib

---

## 📂 Project Structure

```
├── templates/
│   └── home.html
├── app.py
├── train_all.py
├── churn_model.pkl
├── encoder.pkl
├── explainer.pkl
├── hr_employee_churn_data.csv
├── requirements.txt
```

---

## ⚙️ How It Works

1. User enters employee details via web form
2. Data is preprocessed (encoding + formatting)
3. Model predicts churn (Stay/Leave)
4. SHAP explains:

   * Which features influenced the decision
   * How strongly they impacted the output

---

## 📊 SHAP Explanation

* **Base Value** → Average model prediction

* **SHAP Value** → Contribution of each feature

* **Final Output** → Base Value + Sum of SHAP values

* 🔴 Red → Increases likelihood of leaving

* 🟢 Green → Increases likelihood of staying

---

## ▶️ Run the Project

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Train the model (optional)

```
python train_all.py
```

### 3. Run Flask app

```
python app.py
```

### 4. Open in browser

```
http://127.0.0.1:5000/
```

---

## 📈 Model Used

* **XGBoost Classifier**
* Handles tabular data efficiently
* Provides high accuracy and performance

---

## 🎯 Output Example

* Employee is likely to **STAY / LEAVE**
* SHAP graph showing feature contribution
* Base Value & Final Output displayed

---

## 🔮 Future Improvements

* Deploy on AWS / Render / Azure
* Improve UI with dashboards
* Add real-time database integration
* Use deep learning models (ANN)


