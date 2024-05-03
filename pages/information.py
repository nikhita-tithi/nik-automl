#Importing all the necessary libraries and packages
import streamlit as st 
import navbar

#streamlit-wide-layout
st.set_page_config(layout="wide",page_title="AutoML by Nikhita") 

# Setting the navigation bar
current_page = "Data Information"
st.header(current_page)

navbar.nav(current_page)

#----------------- Data Information (Data Summary)--------------------------------------------------------

st.header("Data Information")
df = st.session_state['df']
if df is not None: 
    # show entire data
    data = df
    st.write("Show all data")
    st.write(data)
    st.write('Data Shape: ', df.shape)
    st.write('Data Description')
    st.write(df.describe())

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

    # Going to the next tab
    st.markdown("To Auto-Analyze your data click click on the button below!")
    st.page_link('pages/eda.py', label="Next Tab - Data Analysis")
    
else:
    st.error("Oops looks like your data is unavailable. Please try uploading you data again.")

    # Going to the previous tab
    st.markdown("To go back and re-upload your data click on the button below!")
    st.page_link('Home.py', label="Back - Re-upload Data")