import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Credit Card Fraud Detection Insights",
                   page_icon=":credit_card:",
                   layout="wide")

# Load data
data = pd.read_csv('creditcard.csv')

# Add page title
st.title("Credit Card Fraud Detection Insights")

# Add image
img_url = "https://images.unsplash.com/photo-1501504907352-7c58b82e2d77"
st.image(img_url, width=700)

# Add section headers
st.write("""
## Overview
""")
st.write("""
Credit card fraud is an ongoing problem in the financial industry. According to a report by the Federal Reserve, losses from credit card fraud totaled over $11 billion in 2015. In this project, we aim to develop a machine learning model to detect fraudulent credit card transactions.
""")

st.write("""
## Exploratory Data Analysis
""")
st.write("""
We first start by exploring the dataset to gain insights into the data.
""")
st.write("""
### Distribution of the Target Variable
""")
fig, ax = plt.subplots(figsize=(5,5))
sns.countplot(data['Class'], palette='rocket')
plt.title("Distribution of the Target Variable")
plt.xlabel("Class (0: Non-Fraudulent, 1: Fraudulent)")
plt.ylabel("Count")
st.pyplot(fig)

st.write("""
### Correlation Matrix
""")
corr = data.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    fig, ax = plt.subplots(figsize=(10,10))
    ax = sns.heatmap(corr, mask=mask, square=True, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
st.pyplot(fig)

st.write("""
## Model Comparison
""")
st.write("""
We compare the performance of three machine learning models: logistic regression, decision tree classifier, and random forest classifier.
""")

# Load model results
lr_results = pd.read_csv('lr_results.csv')
dtc_results = pd.read_csv('dtc_results.csv')
rfc_results = pd.read_csv('rfc_results.csv')

# Add accuracy comparison chart
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(lr_results['C'], lr_results['Accuracy'], label='Logistic Regression', marker='o')
ax.plot(dtc_results['Max Depth'], dtc_results['Accuracy'], label='Decision Tree Classifier', marker='o')
ax.plot(rfc_results['N Estimators'], rfc_results['Accuracy'], label='Random Forest Classifier', marker='o')
ax.set_xlabel('Parameter')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Comparison')
ax.legend()
st.pyplot(fig)

# Add feature importance comparison chart
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(dtc_results['Feature'], dtc_results['Importance'], label='Decision Tree Classifier', marker='o')
ax.plot(rfc_results['Feature'], rfc_results['Importance'], label='Random Forest Classifier', marker='o')
ax.set_xlabel('Feature')
ax.set_ylabel('Importance')
ax.set_title('Feature Importance Comparison')
ax.legend()
st.pyplot(fig)

# Add section for conclusions
st.write("""
## Conclusions
""")
st.write("""
Based on our analysis, we found that the random forest classifier outperformed the other two models with an accuracy of 99.95%. Additionally, the most important features for detecting credit
""")