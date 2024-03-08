#Importing all the necessary libraries and packages
import streamlit as st 
import pandas as pd
from streamlit_option_menu import option_menu
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import ClassificationExperiment
from sklearn.datasets import load_diabetes
import os 
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators #for lazypred
from sklearn.base import ClassifierMixin #for lazypred
import matplotlib.pyplot as plt #for lazypred
import seaborn as sns #for lazypred
import base64 #for lazypred
import io #for lazypred
# import time #tpot
# from streamlit_shap import st_shap
# import shap

#streamlit-wide-layout
st.set_page_config(layout="wide") 

# Load the sample file from local folder
if os.path.exists('dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

# Set default session state variables
st.session_state.setdefault('df', None)
# st.session_state.setdefault('target', None)
# st.session_state.setdefault('col_remove', [])
# st.session_state.setdefault('norm_method', None)

# Function to Download CSV data 
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

# Function to download image files of charts
def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

# title of the web application
st.title("Welcome to iBioML - AutoML")

# adding a text description
st.text("This web application allows users to automate their machine learning processes by just uploading their data. ")

# adding a horizontal divider
st.divider()

#adding a horizontal menu
selected = option_menu(
    menu_title=None,
    options=["Home","Data Information", "EDA","Data Cleaning","PyCaret","LazyPredict"],
    icons=["house"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    )

#----------------- HOME (upload or select data)--------------------------------------------------------
if selected == "Home":
    st.header("Upload Your Dataset")
    file = st.file_uploader('Upload your data here', type=['csv']) #leaving it empty as we have already given a header with a label
    st.text("Or you can use this sample dataset:")

    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df) 
        st.text("To Auto-Analyze your data click the next Tab - Data Information!")
        if st.button("Next Tab"):
            selected="Data Information"
    elif(st.button("[Diabetes_dataset.csv](http://nik-automl.streamlit.app/app/static/diabetes_dataset.csv)")):
        if pd.read_csv("http://localhost:8501/app/static/diabetes_dataset.csv") is not None:
            df = pd.read_csv("http://localhost:8501/app/static/diabetes_dataset.csv")
            df.to_csv('dataset.csv', index=None)
        else:
            diabetes = load_diabetes()
            X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
            Y = pd.Series(diabetes.target, name='response')
            df = pd.concat( [X,Y], axis=1 )
        st.write('Data Uploaded Successfully')
        st.text("To Auto-Analyze your data click next!")
        if st.button("Next Tab"):
            selected="Data Information"
    else:
        st.error("Please Upload a .csv file!")


#----------------- Data Information (Data Summary)--------------------------------------------------------
if selected == "Data Information":
    st.header("Data Information")
    
    if df is not None: 
        # show entire data
        data = df
        st.write("Show all data")
        st.write(data)
        st.write('Data Shape: ', df.shape)
        st.write('Data Description')
        st.write(df.describe())
        st.session_state['df'] = df

        # show column names
        st.write("Show Column Names")
        st.write(data.columns)

        # show dimensions
        st.write("Show Dimensions")
        st.write(data.shape)
     
        # show summary
        st.write("Show Summary")
        st.write(data.describe())
    
        # show missing values
        st.write("Show Missing Values")
        st.write(data.isna().sum())
        
    else:
        st.__loader__
 # else:
 #     st.error("Oops looks like your data is unavailable. Please try uploading you data again.")
        
#----------------- Data Cleaning (Data Preprocessing)--------------------------------------------------------
if selected == "Data Cleaning":
    st.header("Data Cleaning")
    
    if df is not None: 
        st.dataframe(df)
        #Add data cleaning code here
    else:
        st.__loader__
 # else:
 #     st.error("Oops looks like your data is unavailable. Please try uploading you data again.")



#----------------- EDA (Data analysis)--------------------------------------------------------
if selected == "EDA":
    st.header("Auto-analysis - EDA")
    profile = ProfileReport(df, title="Profiling Report")
    if profile: 
        st_profile_report(profile)
        if st.button("Home"):
            selected="Home"
        if st.button("AutoML"):
            selected="AutoML"
    else:
        st.__loader__
 # else:
 #     st.error("Oops looks like your data is unavailable. Please try uploading you data again.")
