import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import io
import base64
import matplotlib

matplotlib.use('Agg')

app = Flask(__name__)

# Load models
churn_model = pickle.load(open('churn_model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
explainer = pickle.load(open('explainer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    df = pd.DataFrame([data])

    salary_encoded = encoder.transform(df[['salary']])
    salary_df = pd.DataFrame(
        salary_encoded,
        columns=encoder.get_feature_names_out(['salary'])
    )

    df_numeric = df.drop(['salary', 'empid'], axis=1, errors='ignore')
    final_df = pd.concat([df_numeric, salary_df], axis=1)

    output = churn_model.predict(final_df)
    return jsonify(int(output[0]))


@app.route('/predict', methods=['POST'])
def predict():

    form_data = request.form.to_dict()
    df = pd.DataFrame([form_data])

    numeric_features = [
        'satisfaction_level', 'last_evaluation', 'number_project',
        'average_montly_hours', 'time_spend_company',
        'Work_accident', 'promotion_last_5years'
    ]

    df[numeric_features] = df[numeric_features].astype(float)

    salary_encoded = encoder.transform(df[['salary']])
    salary_df = pd.DataFrame(
        salary_encoded,
        columns=encoder.get_feature_names_out(['salary'])
    )

    final_input = pd.concat([df[numeric_features], salary_df], axis=1)

    prediction = churn_model.predict(final_input)[0]

    # SHAP VALUES
    shap_values = explainer(final_input)

    values = shap_values.values[0]
    features = final_input.columns

    idx = np.argsort(values)
    sorted_values = values[idx]
    sorted_features = features[idx]

    colors = ['red' if v > 0 else 'green' for v in sorted_values]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_values, color=colors)
    for i, v in enumerate(sorted_values):

        if v > 0:
         plt.text(v + 0.02, i, f"{round(v,2)}", va='center')
        else:
          plt.text(v - 0.02, i, f"{round(v,2)}", va='center')
    plt.axvline(0, linestyle='--')

    # Updated labels
    plt.xlabel("SHAP Value (Impact on Model Output)")
    plt.ylabel("Employee Features")
    plt.title("Feature Contribution to Employee Churn")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()

    plot_url = base64.b64encode(buf.getvalue()).decode()

    # Base & final values
    base_value = explainer.expected_value
    final_value = base_value + sum(values)

    if prediction == 1:
        display_text = "Employee is likely to LEAVE"
    else:
        display_text = "Employee is likely to STAY"

    return render_template(
        'home.html',
        prediction_text=display_text,
        shap_plot=plot_url,
        base_value=round(base_value, 2),
        final_value=round(final_value, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)