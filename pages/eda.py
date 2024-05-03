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

if 'df' in st.session_state and st.session_state['df'] is not None: 
    df = st.session_state['df']
    low_level = False
    if len(df.columns)>55:
        low_level = True
    profile = ProfileReport(df, title="Profiling Report",explorative=True,minimal=low_level)
    if profile: 
        st_profile_report(profile)

        # Going to the next tab
        st.markdown("To clean your data click on the button below!")
        st.page_link('pages/cleaning.py', label="Next Tab - Data Cleaning")
    else:
        st.__loader__
else:
     st.error("Oops looks like your data is unavailable. Please try uploading you data again.")

    # Going to the previous tab
     st.markdown("To go back and re-upload your data click on the button below!")
     st.page_link('Home.py', label="Back - Re-upload Data")