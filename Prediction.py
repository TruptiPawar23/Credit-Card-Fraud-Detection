import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

st.write("# Credit Card Fraud Detection")

st.sidebar.header('Input Credit Card Details')

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=['csv'])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input():
        V1 = st.sidebar.slider('V1', -5.0, 1.5, 5.0)
        V2 = st.sidebar.slider('V2', -5.0, 1.5, 5.0)
        V3 = st.sidebar.slider('V3', -5.0, 1.5, 5.0)
        V4 = st.sidebar.slider('V4', -5.0, 1.5, 5.0)
        V5 = st.sidebar.slider('V5', -5.0, 1.5, 5.0)
        V6 = st.sidebar.slider('V6', -5.0, 1.5, 5.0)
        V7 = st.sidebar.slider('V7', -5.0, 1.5, 5.0)
        V8 = st.sidebar.slider('V8', -5.0, 1.5, 5.0)
        V9 = st.sidebar.slider('V9', -5.0, 1.5, 5.0)
        V10 = st.sidebar.slider('V10', -5.0, 1.5, 5.0)
        V11 = st.sidebar.slider('V11', -5.0, 1.5, 5.0)
        V12 = st.sidebar.slider('V12', -5.0, 1.5, 5.0)
        V13 = st.sidebar.slider('V13', -5.0, 1.5, 5.0)
        V14 = st.sidebar.slider('V14', -5.0, 1.5, 5.0)
        V15 = st.sidebar.slider('V15', -5.0, 1.5, 5.0)
        V16 = st.sidebar.slider('V16', -5.0, 1.5, 5.0)
        V17 = st.sidebar.slider('V17', -5.0, 1.5, 5.0)
        V18 = st.sidebar.slider('V18', -5.0, 1.5, 5.0)
        V19 = st.sidebar.slider('V19', -5.0, 1.5, 5.0)
        V20 = st.sidebar.slider('V20', -5.0, 1.5, 5.0)
        V21 = st.sidebar.slider('V21', -5.0, 1.5, 5.0)
        V22 = st.sidebar.slider('V22', -5.0, 1.5, 5.0)
        V23 = st.sidebar.slider('V23', -5.0, 1.5, 5.0)
        V24 = st.sidebar.slider('V24', -5.0, 1.5, 5.0)
        V25 = st.sidebar.slider('V25', -5.0, 1.5, 5.0)
        V26 = st.sidebar.slider('V26', -5.0, 1.5, 5.0)
        V27 = st.sidebar.slider('V27', -5.0, 1.5, 5.0)
        V28 = st.sidebar.slider('V28', -5.0, 1.5, 5.0)
        Amount = st.sidebar.number_input('Amount')

        data = {'V1': V1,
                'V2': V2,
                'V3': V3,
                'V4': V4,
                'V5': V5,
                'V6': V6,
                'V7': V7,
                'V8': V8,
                'V9': V9,
                'V10': V10,
                'V11': V11,
                'V12': V12,
                'V13': V13,
                'V14': V14,
                'V15': V15,
                'V16': V16,
                'V17': V17,
                'V18': V18,
                'V19': V19,
                'V20': V20,
                'V21': V21,
                'V22': V22,
                'V23': V23,
                'V24': V24,
                'V25': V25,
                'V26': V26,
                'V27': V27,
                'V28': V28,
                'Amount': Amount
                }
        fea = pd.DataFrame(data, index=[0])
        return fea


    input_df = user_input()



st.subheader('Credit Card Data')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded.')
    st.write(input_df)

load_clf = joblib.load(open('savedModels/model.joblib', 'rb'))

prediction = load_clf.predict(input_df)
prediction_probability = load_clf.predict_proba(input_df)

st.subheader('Prediction')
if prediction == 0:
    st.write("Genuine Transaction")
else:
    st.write("Fraudulent Transaction")

st.subheader('Prediction Probability')
st.write(prediction_probability)
