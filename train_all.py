import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import shap

# ==============================
# STEP 1: LOAD DATA
# ==============================
df = pd.read_csv('hr_employee_churn_data.csv')

# ==============================
# STEP 2: CLEAN DATA
# ==============================
df2 = df.copy()
df2.drop(['empid'], axis=1, inplace=True)

# Fill missing values
df2['satisfaction_level'] = df2['satisfaction_level'].fillna(
    df2['satisfaction_level'].mean()
)

# ==============================
# STEP 3: ENCODING
# ==============================
encoder = OneHotEncoder(drop='first', sparse_output=False)
salary_encoded = encoder.fit_transform(df2[['salary']])

salary_df = pd.DataFrame(
    salary_encoded,
    columns=encoder.get_feature_names_out(['salary'])
)

df2 = pd.concat([df2.drop('salary', axis=1), salary_df], axis=1)

# Save encoder
pickle.dump(encoder, open('encoder.pkl', 'wb'))

# ==============================
# STEP 4: SPLIT DATA
# ==============================
X = df2.drop('left', axis=1)
y = df2['left']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ==============================
# STEP 5: TRAIN MODEL
# ==============================
model = XGBClassifier(
    objective='binary:logistic',
    learning_rate=0.1,
    max_depth=20,
    n_estimators=200
)

model.fit(X_train, y_train)

print("Model Accuracy:", model.score(X_test, y_test))

# Save model
pickle.dump(model, open('churn_model.pkl', 'wb'))

# ==============================
# STEP 6: CREATE SHAP EXPLAINER
# ==============================
X_sample = X_train.sample(100, random_state=42)

explainer = shap.Explainer(model, X_sample)

# Save explainer
pickle.dump(explainer, open('explainer.pkl', 'wb'))

print("Everything saved successfully!")