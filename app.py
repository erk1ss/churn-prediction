import streamlit as st
import pandas as pd
import joblib

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title('Predicție Abandon Client (Customer Churn)')
st.write('Introduceți datele clientului pentru a estima riscul de abandon.')
st.header('Date Client')

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider('Tenure (luni)', 0, 72, 12)
    monthly_charges = st.slider('Factura lunară ($)', 18.0, 120.0, 65.0)
    contract = st.selectbox('Tip contract', ['Month-to-month', 'One year', 'Two year'])
    internet_service = st.selectbox('Tip internet', ['DSL', 'Fiber optic', 'No'])
    payment_method = st.selectbox('Metodă de plată', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

with col2:
    gender = st.selectbox('Gen', ['Male', 'Female'])
    senior_citizen = st.selectbox('Vârstnic (Senior Citizen)', [0, 1])
    partner = st.selectbox('Are partener', ['Yes', 'No'])
    dependents = st.selectbox('Are persoane în întreținere', ['Yes', 'No'])
    paperless_billing = st.selectbox('Facturare electronică', ['Yes', 'No'])

if st.button('Prezice riscul de abandon'):
    
    input_dict = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'SeniorCitizen': senior_citizen,
        'gender_Male': 1 if gender == 'Male' else 0,
        'Partner_Yes': 1 if partner == 'Yes' else 0,
        'Dependents_Yes': 1 if dependents == 'Yes' else 0,
        'PaperlessBilling_Yes': 1 if paperless_billing == 'Yes' else 0,
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
        'InternetService_No': 1 if internet_service == 'No' else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
    }
    
    input_df = pd.DataFrame([input_dict])
    
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[model_columns]
    
    input_df[['tenure', 'MonthlyCharges']] = scaler.transform(input_df[['tenure', 'MonthlyCharges']])
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    if prediction == 1:
        st.error(f'⚠️ Risc ridicat de abandon: {probability*100:.1f}%')
    else:
        st.success(f'✅ Risc scăzut de abandon: {probability*100:.1f}%')    
    
    st.subheader('Factori care au influențat predicția')
    
    contributions = input_df.iloc[0] * model.coef_[0]
    contributions_df = pd.DataFrame({
        'Factor': input_df.columns,
        'Contribuție': contributions.values
    })
    contributions_df = contributions_df[contributions_df['Contribuție'] != 0]
    contributions_df = contributions_df.sort_values('Contribuție', ascending=False).head(10)
    
    st.bar_chart(contributions_df.set_index('Factor'))