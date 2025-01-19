# end to end model py file
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder

# load the model
with open('rfmodel.pkl','rb') as file:
    model = pickle.load(file)


# load the encoders
with open('ohe_encoder.pkl','rb') as file:
    ohe_department = pickle.load(file)

with open('data_scaler.pkl','rb') as file:
    scaler = pickle.load(file)

with open('Ordinal_encoder_salary.pkl','rb') as file:
    oe_salary = pickle.load(file)

# using streamlit app
st.title('Employee churn random forest model')

# User input
satisfaction_level  = st.number_input("satisfaction level on scale 0-1",min_value=0.1,max_value=1.0)
last_evaluation = st.number_input("last evaluated score",min_value=0.3,max_value=1.0)
number_project = st.slider('Number of Projects', 2, 7)
average_monthly_hours = st.number_input("monthly working hours",min_value=95,max_value=300)
tenure = st.slider('tenure', 2, 5)
work_accident = st.selectbox('work accident', [0, 1])
promotion_last_5years = st.selectbox('promotion_last_5years', [0, 1])
salary = st.selectbox('salary', ["low","medium","high"])
department = st.selectbox('department', ["sales","technical","support","IT","RandD","product_mng","marketing","accounting","hr","management"])

# Prepare the input data
input_data = pd.DataFrame({
    'satisfaction_level': [satisfaction_level],
    'last_evaluation': [last_evaluation],
    'number_project': [number_project],
    'average_monthly_hours': [average_monthly_hours],
    'tenure': [tenure],
    'work_accident': [work_accident],
    'promotion_last_5years': [promotion_last_5years],
    'salary': [oe_salary.transform([[salary]])+1.0],
})

# One-hot encode 'Geography'
dept_encoded = ohe_department.transform([[department]])
dept_encoded_df = pd.DataFrame(dept_encoded, columns=ohe_department.get_feature_names_out(['department']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), dept_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict_proba(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The employee is not in risk.')
else:
    st.write('The employee is in risk.')
    