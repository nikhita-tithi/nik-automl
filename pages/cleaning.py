#Importing all the necessary libraries and packages
import streamlit as st 
import navbar
import AutoDataCleaner.AutoDataCleaner as adc

#streamlit-wide-layout
st.set_page_config(layout="wide",page_title="AutoML by Nikhita") 

# Setting the navigation bar
current_page = "Data Cleaning"
st.header(current_page)

navbar.nav(current_page)

# Method to display the cleaned data
def display_data(cleaned_data):
    st.write("After cleaning the data...")
    st.write(cleaned_data)
    # Going to the previous tab
    st.markdown("To go back and re-upload your data click on the button below!")
    st.page_link('Home.py', label="Back - Re-upload Data")

#----------------- Data Cleaning (Data Preprocessing)--------------------------------------------------------

st.header("Data Cleaning")

if 'df' in st.session_state and st.session_state['df'] is not None: 
    # Fetching session state variables
    df = st.session_state['df']
    cleaned_data = st.session_state['cleaned_data']

    # mode = st.selectbox('Select the mode:', ['auto','manual'])
    detect_binary = st.selectbox('Do you want to detect binary?', [False,True])
    # duplicates = st.selectbox('Select the duplicates:', [False,'auto','True'])
    # missing_num = st.selectbox('Select the Missing number:', [False,'auto', 'linreg', 'knn', 'mean', 'median', 'most_frequent', 'delete'])
    missing_num = st.selectbox('Select the Missing number:', [False,'remove row', 'mean', 'mode', '*'])
    numeric_dtype = st.selectbox('Convert data to numeric values?', [True,False])
    one_hot = st.selectbox('Implement one-hot encoding?:', [True,False])
    normalize = st.selectbox('Normalize?:', [True,False])
    remove_columns = st.multiselect("Remove columns, if any:", df.columns)
    # missing_categ = st.selectbox('Select the missing categorical:', [False,'auto', 'logreg', 'knn', 'most_frequent', 'delete'])
    # encode_categ = st.selectbox('Select the Encoding categorical columns', [False,'auto', ['onehot'], ['label']])
    # outliers = st.selectbox('Select the Outlier:', [False,'auto', 'winz', 'delete'])
    verbose = st.selectbox('Display logs (verbose):', [True,False])
    st.dataframe(df)
    data = df
    if st.button("Run Cleaning"):
    
        st.write("After cleaning the data...")
        new_data = adc.clean_me(data, detect_binary = detect_binary, one_hot = one_hot, na_cleaner_mode=missing_num, normalize=normalize, remove_columns=remove_columns, verbose=verbose)
        st.write(new_data)
        st.session_state['cleaned_data']=new_data
        # To download the new data
        st.markdown("To download the new data click on the download button below!")
        st.text("*PLEASE NOTE : To use the new data for analysis on previous tabs, you would have to download it and upload the file as new data.*")
        
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(new_data)

        st.download_button(
        "Press to Download",
        csv,
        "new_data.csv",
        "text/csv",
        key='download-csv')
    if cleaned_data is not None:
        display_data(cleaned_data)

        # Going to the next tab
        st.markdown("To run AutoML on your data click click on the button below!")
        st.page_link('pages/automl.py', label="Next Tab - AutoML")
else:
     st.error("Oops looks like your data is unavailable. Please try uploading you data again.")

    # Going to the previous tab
     st.markdown("To go back and re-upload your data click on the button below!")
     st.page_link('Home.py', label="Back - Re-upload Data")

