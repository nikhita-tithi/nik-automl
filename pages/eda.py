#Importing all the necessary libraries and packages
import streamlit as st 
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
import navbar

#streamlit-wide-layout
st.set_page_config(layout="wide",page_title="AutoML by Nikhita") 

# Setting the navigation bar
current_page = "Data Analysis"
st.header(current_page)

navbar.nav(current_page)

#----------------- EDA (Data analysis)--------------------------------------------------------
st.header("Auto-analysis - EDA")
df = st.session_state['df']

if df is not None:

    low_level = False
    if len(df.columns)>55:
        low_level = True
    profile = ProfileReport(df, title="Profiling Report",explorative=True,minimal=low_level)
    if profile: 
        st_profile_report(profile)

        # Going to the next tab
        st.markdown("To run AutoML on your data click click on the button below!")
        st.page_link('pages/automl.py', label="Next Tab - AutoML")
    else:
        st.__loader__
else:
     st.error("Oops looks like your data is unavailable. Please try uploading you data again.")

    # Going to the previous tab
     st.markdown("To go back and re-upload your data click on the button below!")
     st.page_link('Home.py', label="Back - Re-upload Data")