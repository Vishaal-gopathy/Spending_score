#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_excel('family_financial_and_transactions_data.xlsx')


# In[3]:


df


# In[4]:


df.head()


# In[5]:


import numpy as np

def clean_data(df):
    
    df=df
    # Data cleaning steps
    # Convert date column to datetime
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    
    # Handle missing values
    df.fillna({
        'Amount': df['Amount'].median(),
        'Income': df['Income'].median(),
        'Savings': df['Savings'].median(),
        'Monthly Expenses': df['Monthly Expenses'].median()
    }, inplace=True)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def aggregate_family_data(df):
   
    family_aggregation = df.groupby('Family ID').agg({
        'Amount': 'sum',
        'Income': 'first',
        'Savings': 'first',
        'Monthly Expenses': 'first',
        'Loan Payments': 'first',
        'Credit Card Spending': 'first',
        'Dependents': 'first',
        'Financial Goals Met (%)': 'first',
        'Category': lambda x: x.value_counts().index[0]
    }).reset_index()
    
    return family_aggregation

def calculate_financial_ratios(df):
 
    df['Savings_to_Income_Ratio'] = df['Savings'] / df['Income'] * 100
    df['Monthly_Expenses_Ratio'] = df['Monthly Expenses'] / df['Income'] * 100
    df['Loan_Payment_Ratio'] = df['Loan Payments'] / df['Income'] * 100
    df['Credit_Card_Spending_Ratio'] = df['Credit Card Spending'] / df['Income'] * 100
    
    return df


# In[6]:


clean_data(df)


# In[7]:


df=aggregate_family_data(df)
df


# In[8]:


calculate_financial_ratios(df)


# In[9]:


class FinancialScoreCalculator:
    def __init__(self):
        # Define weights for different financial factors
        self.weights = {
            'savings_ratio': 0.25,
            'expenses_ratio': 0.20,
            'loan_payment_ratio': 0.15,
            'credit_card_ratio': 0.10,
            'financial_goals': 0.15,
            'dependents_factor': 0.15
        }
    
    def calculate_score(self, family_data):
        # Savings to Income Ratio Score (lower is better)
        savings_score = max(0, 100 - family_data['Savings_to_Income_Ratio'])
        
        # Monthly Expenses Ratio Score (lower is better)
        expenses_score = max(0, 100 - family_data['Monthly_Expenses_Ratio'])
        
        # Loan Payment Ratio Score (lower is better)
        loan_score = max(0, 100 - family_data['Loan_Payment_Ratio'])
        
        # Credit Card Spending Score (lower is better)
        credit_card_score = max(0, 100 - family_data['Credit_Card_Spending_Ratio'])
        
        # Financial Goals Score
        goals_score = family_data['Financial Goals Met (%)']
        
        # Dependents Factor (adjust score based on number of dependents)
        dependents_factor = max(50, 100 - (family_data['Dependents'] * 10))
        
        # Weighted Score Calculation
        final_score = (
            (savings_score * self.weights['savings_ratio']) +
            (expenses_score * self.weights['expenses_ratio']) +
            (loan_score * self.weights['loan_payment_ratio']) +
            (credit_card_score * self.weights['credit_card_ratio']) +
            (goals_score * self.weights['financial_goals']) +
            (dependents_factor * self.weights['dependents_factor'])
        )
        
        return round(final_score, 2)
    
    def generate_recommendations(self, family_data, score):
    
        recommendations = []

        # Score-based overall recommendation
        if score < 40:
            recommendations.append("Critical financial health. Immediate comprehensive financial planning needed.")
        elif score < 60:
            recommendations.append("Your financial health needs significant improvement.")
        elif score < 75:
            recommendations.append("You're on the right track, but there's room for improvement.")

        # Specific metric-based recommendations
        if family_data['Savings_to_Income_Ratio'] < 20:
            recommendations.append(f"Increase savings. Currently saving only {family_data['Savings_to_Income_Ratio']:.2f}% of income.")

        if family_data['Monthly_Expenses_Ratio'] > 60:
            recommendations.append(f"Reduce monthly expenses. Currently spending {family_data['Monthly_Expenses_Ratio']:.2f}% of income.")

        if family_data['Loan_Payment_Ratio'] > 30:
            recommendations.append(f"Debt management needed. Loan payments are {family_data['Loan_Payment_Ratio']:.2f}% of income.")

        if family_data['Credit_Card_Spending_Ratio'] > 20:
            recommendations.append(f"Control credit card spending. Currently at {family_data['Credit_Card_Spending_Ratio']:.2f}% of income.")

        if family_data['Financial Goals Met (%)'] < 50:
            recommendations.append(f"Goal achievement low. Currently meeting only {family_data['Financial Goals Met (%)']:.2f}% of financial goals.")

        # Personalized improvement suggestions based on score
        if score < 50:
            recommendations.append("Consider creating a strict budget and tracking expenses closely.")
            recommendations.append("Explore additional income streams or part-time work.")

        if score < 70:
            recommendations.append("Start an emergency fund if you haven't already.")
            recommendations.append("Review and potentially reduce unnecessary subscriptions and expenses.")

        return recommendations


# In[10]:


calc=FinancialScoreCalculator()


# In[11]:


# Prepare the dataframe
df= calculate_financial_ratios(df)

# Calculate scores for each family
df['Financial_Score'] = df.apply(calc.calculate_score, axis=1)


# In[12]:


df


# In[13]:


# Generate recommendations for each family
df['Recommendations'] = df.apply(
    lambda row: calc.generate_recommendations(row, row['Financial_Score']), 
    axis=1
)


# In[14]:


df


# 
# 
# 1. Score-Based Overall Recommendations:
# 
# if score < 40:
#     recommendations.append("Critical financial health. Immediate comprehensive financial planning needed.")
# 
# - Justification: 
#   - Scores below 40 indicate severe financial distress
#   - Suggests urgent intervention is required
#   - Signals potential risk of financial collapse or bankruptcy
#   - Implies multiple financial metrics are performing poorly
# 
# elif score < 60:
#     recommendations.append("Your financial health needs significant improvement.")
# 
# - Justification:
#   - Scores between 40-59 suggest moderate financial challenges
#   - Indicates systemic issues in financial management
#   - Requires comprehensive financial restructuring
#   - Highlights need for immediate corrective actions
# 
# 
# elif score < 75:
#     recommendations.append("You're on the right track, but there's room for improvement.")
# 
# - Justification:
#   - Scores 60-74 represent average financial health
#   - Acknowledges some positive financial behaviors
#   - Encourages continued financial discipline
#   - Suggests potential for optimization
# 
# 2. Savings-to-Income Ratio Recommendation:
# 
# if family_data['Savings_to_Income_Ratio'] < 20:
#     recommendations.append(f"Increase savings. Currently saving only {family_data['Savings_to_Income_Ratio']:.2f}% of income.")
# 
# - Justification:
#   - Financial experts recommend saving 20-30% of income
#   - Less than 20% savings indicates financial vulnerability
#   - Insufficient emergency fund protection
#   - Limits future financial flexibility and investment opportunities
# 
# 3. Monthly Expenses Recommendation:
# 
# if family_data['Monthly_Expenses_Ratio'] > 60:
#     recommendations.append(f"Reduce monthly expenses. Currently spending {family_data['Monthly_Expenses_Ratio']:.2f}% of income.")
# 
# - Justification:
#   - Standard financial advice suggests keeping expenses under 50-60% of income
#   - Over 60% expenses indicate potential financial strain
#   - Leaves little room for savings, investments, or unexpected costs
#   - High risk of living paycheck to paycheck
# 
# 4. Loan Payment Recommendation:
# 
# if family_data['Loan_Payment_Ratio'] > 30:
#     recommendations.append(f"Debt management needed. Loan payments are {family_data['Loan_Payment_Ratio']:.2f}% of income.")
# 
# - Justification:
#   - Financial experts recommend keeping debt payments under 30% of income
#   - Higher ratios suggest over-leveraging
#   - Increases financial stress and risk
#   - Limits ability to save or invest
#   - Potential credit score impact
# 
# 5. Credit Card Spending Recommendation:
# 
# if family_data['Credit_Card_Spending_Ratio'] > 20:
#     recommendations.append(f"Control credit card spending. Currently at {family_data['Credit_Card_Spending_Ratio']:.2f}% of income.")
# 
# - Justification:
#   - Credit card spending over 20% of income is a red flag
#   - High risk of accumulating high-interest debt
#   - Suggests potential financial indiscipline
#   - Increases overall financial vulnerability
# 
# 6. Financial Goals Recommendation:
# 
# if family_data['Financial Goals Met (%)'] < 50:
#     recommendations.append(f"Goal achievement low. Currently meeting only {family_data['Financial Goals Met (%)']:.2f}% of financial goals.")
# 
# - Justification:
#   - Less than 50% goal achievement indicates poor financial planning
#   - Suggests misalignment between financial strategies and objectives
#   - Requires reassessment of financial goals and methods
# 
# 7. Low Score-Based Personalized Recommendations:
# 
# if score < 50:
#     recommendations.append("Consider creating a strict budget and tracking expenses closely.")
#     recommendations.append("Explore additional income streams or part-time work.")
# 
# - Justification:
#   - Scores below 50 indicate comprehensive financial challenges
#   - Suggests need for strict financial discipline
#   - Recommends proactive income enhancement
#   - Focuses on both expense reduction and income increase
# 
# 
# if score < 70:
#     recommendations.append("Start an emergency fund if you haven't already.")
#     recommendations.append("Review and potentially reduce unnecessary subscriptions and expenses.")
# 
# - Justification:
#   - Scores 50-69 need continued financial improvement
#   - Emergency fund crucial for financial stability
#   - Unnecessary expenses can be significant financial drain
#   - Emphasizes building financial resilience

# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns

class FinancialInsightVisualizer:
    def __init__(self, data):
        self.data = data
    
    def spending_distribution(self):
        
        plt.figure(figsize=(10, 6))
        category_spending = self.data.groupby('Category')['Amount'].sum()
        category_spending.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Spending Distribution Across Categories')
        plt.xlabel('Category')
        plt.ylabel('Total Spending')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show() 
    
    def financial_scores_boxplot(self):
        
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=self.data['Financial_Score'])
        plt.title('Distribution of Financial Health Scores')
        plt.xlabel('Financial Health Score')
        plt.tight_layout()
        plt.show() 


# In[16]:


visual=FinancialInsightVisualizer(df)


# In[17]:


visual.spending_distribution()


# In[18]:


visual.financial_scores_boxplot()


# In[21]:


# Import necessary libraries
from flask import Flask, request, jsonify
import pandas as pd
import threading

# Create a Flask app
app = Flask(__name__)

# Define your data processing functions here
def clean_data(df):
    # Example cleaning function
    df.fillna(0, inplace=True)
    return df

@app.route('/score', methods=['POST'])
def score():
    family_data = request.json
    df = pd.DataFrame(family_data)

    # Clean and preprocess data
    df = clean_data(df)

    # Dummy scoring logic for illustration
    df['Financial Score'] = df['Income'] / (df['Monthly Expenses'] + 1) * 100  # Simple scoring example
    results = df.to_dict(orient='records')

    return jsonify(results)

# Function to run the Flask app
def run_flask():
    app.run(port=5000)

# Start the Flask app in a separate thread
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()


# In[22]:


import streamlit as st
import requests

# Function to run the Streamlit app
def run_streamlit():
    st.title("Financial Insights Dashboard")

    # Input fields for family data
    family_id = st.text_input("Family ID")
    income = st.number_input("Monthly Income", min_value=0)
    savings = st.number_input("Total Savings", min_value=0)
    monthly_expenses = st.number_input("Monthly Expenses", min_value=0)

    if st.button("Calculate Score"):
        family_data = [{
            "Family ID": family_id,
            "Income": income,
            "Savings": savings,
            "Monthly Expenses": monthly_expenses
        }]
        
        # Call the API
        response = requests.post("http://127.0.0.1:5000/score", json=family_data)
        result = response.json()
        
        st.write("Financial Scores:")
        for res in result:
            st.write(f"Family ID: {res['Family ID']}, Financial Score: {res['Financial Score']}")

# Start the Streamlit app in a separate thread
streamlit_thread = threading.Thread(target=run_streamlit)
streamlit_thread.start()


# In[ ]:




