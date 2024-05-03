#Importing all the necessary libraries and packages
import os 
import streamlit as st 
import pandas as pd
import navbar

#streamlit-wide-layout
st.set_page_config(layout="wide",page_title="AutoML by Nikhita") 

# Load the sample file from local folder
if os.path.exists('dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

# title of the web application
st.title("Welcome to iBioML - AutoML")

# Display the menu
navbar.nav('Home')

# adding a text description
st.markdown("This application allows users to automate their machine learning processes by just uploading their data. ")

st.markdown("""
            This app helps the user to Analyze, Clean, Model & Interpret their data through the various modules as below-.
            
            Data Information - This modules gives the basic description and outllook of the uploaded data.
            
            Data Analysis - The module performs the explorative data analysis on your data and helps you analyze and understand your data attributes(features) better. The alert tab in this also highlights any data imbalances and missing values.
            
            Data Cleaning - This modules helps you clean your data, If you like the cleaned result, you can save this data and re-upload it to analyze and process the new data.
            
            AutoML - This is where magic happens! Automatic machine learning is applied on your data to test it with different models.
            A comparative result of the models and the best model are obtained.
            
            Interpret - This modules helps you interpret your best model with interpretable machine learning.

            *Please note - The application results are much better for smaller data, few features are limited for larger datasets.*

            Just play around and have fun!!!
            """)

#----------------- HOME (upload or select data)--------------------------------------------------------
st.header("Upload Your Dataset")
file = st.file_uploader('Upload your data here', type=['csv']) #leaving it empty as we have already given a header with a label
st.markdown("Or you can use this sample dataset:")

# if a file is uploaded
if file: 
    df = pd.read_csv(file, index_col=None)
    df.to_csv('dataset.csv', index=None)
    st.success('Data Uploaded Successfully')
    st.dataframe(df) 
    st.markdown("To Auto-Analyze your data click the next Tab - Data Information!")
    st.page_link('pages/information.py', label="Next Tab - Data Information")

# if existing sample file is used
elif(st.button("[Diabetes_dataset.csv](http://nik-automl.streamlit.app/app/static/diabetes_dataset.csv)")):
    if pd.read_csv("http://localhost:8501/app/static/diabetes_dataset.csv") is not None:
        df = pd.read_csv("http://localhost:8501/app/static/diabetes_dataset.csv")
        df.to_csv('dataset.csv', index=None)
    st.success('Data Uploaded Successfully')
    st.dataframe(df) 
    st.markdown("To Auto-Analyze your data click next!")
    st.page_link('pages/information.py', label="Next Tab - Data Information")

# if no or wrong format file is uploaded 
else:
    st.error("Please Upload a .csv file!")