import streamlit as st
import pandas as pd 
import numpy as np 
import seaborn as sns
import sklearn as sk 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


st.title("LinkedIn User Prediction App")
st.write(f"Hello! Welcome to this app that will predict if you are likely to be a LinkedIn user or not based on some basic information about you!")
name =st.text_input(f"What is your name?")
if name:
    st.write(f"{name} is a nice name!")


def main():
    # Data Loading
    s = pd.read_csv("social_media_usage.csv")
    def clean_sm(x): return np.where(x == 1, 1, 0)
    ss_data = {"sm_li" : s["web1h"],
           "income" : s["income"],
           "education" : s["educ2"],
           "parent" : s["par"],
           "married" : s["marital"],
           "female" : s["gender"],
           "age": s["age"]}
    ss = pd.DataFrame(ss_data)
    ss["sm_li"] = ss["sm_li"].apply(clean_sm)
    ss["female"] = np.where(ss["female"] == 2, 1, 0)
    ss["married"] = np.where(ss["married"] == 1, 1, 0)
    ss["parent"] = np.where(ss["parent"] == 1, 1, 0)
    ss["income"] = np.where((ss["income"] > 9), np.nan, ss["income"])
    ss["age"] = np.where((ss["age"] > 98), np.nan, ss["age"])
    ss["education"] = np.where((ss["education"] > 8), np.nan, ss["education"])
    ss.dropna(subset=["income", "age", "education"], inplace=True)

    # User input for features
    income = st.slider(f"**What is your Household Income?**", 1, 9)
    st.caption(f"1 - Less than $10,000")
    st.caption(f"2 - 10 to under $20,000")
    st.caption(f"3 - 20 to under $30,000")
    st.caption(f"4 - 30 to under $40,000")
    st.caption(f"5 - 40 to under $50,000")
    st.caption(f"6 - 50 to under $75,000")
    st.caption(f"7 - 75 to under $100,000")
    st.caption(f"8 - 100 to under $150,000")
    st.caption(f"9 - $150,000 or more?")

    education = st.slider(f"**What is your level of eduction?**", 1, 8)
    st.caption(f"1 - Less than high school")
    st.caption(f"2 - High school incomplete")
    st.caption(f"3 - High school graduate")
    st.caption(f"4 - Some college, no degree")
    st.caption(f"5 - Two-year associate degree")
    st.caption(f"6 - Bachelor’s degree")
    st.caption(f"7 - Some postgraduate or professional schooling")
    st.caption(f"8 - Postgraduate or professional degree")

    age = st.slider(f"**What is your Age?**", 0, 98)
    st.markdown(f"#### Please check the box next to each option if true:")
    parent = st.checkbox(f"**I am a Parent**")
    married = st.checkbox(f"**I am Married**")
    female = st.checkbox(f"**I am a Female**")
    

    # Make a prediction using the model
    y = ss["sm_li"]
    x = ss.drop(columns=["sm_li"])
    model = LogisticRegression(class_weight="balanced")
    input_data = pd.DataFrame({'income': [income], 'education': [education],
                                'parent': [int(parent)], 'married': [int(married)],
                                'female': [int(female)], 'age': [age]})
    model.fit(x, y)
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]
    
    st.divider()
    
    # Display the prediction and probability
    st.subheader("Prediction:")
    st.write(f"You are likely a LinkedIn user." if prediction == 1 else "You are likely not a LinkedIn user.")
    
    st.subheader("Probability:")
    st.write(f"The probability of being a LinkedIn user is: {probability}")

if __name__ == "__main__":
    main()
