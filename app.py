import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# title
st.title("Welcome to iBioML - AutoML")

# adding a horizontal menu
selected = option_menu(
    menu_title=None,
    options=["Home","EDA","AutoML","Results"],
    icons=["house"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    )

st.header("Upload Your Dataset")
file = st.file_uploader("") #leaving it empty as we have already given a header with a label
if file: 
    df = pd.read_csv(file, index_col=None)
    df.to_csv('dataset.csv', index=None)
    st.dataframe(df) 
    st.header("Auto-analysis - EDA")
    # st.button("Auto-Analysis","EDA")
    profile = ProfileReport(df, title="Profiling Report")
    if profile: 
        st_profile_report(profile)
    else:
        st.__loader__