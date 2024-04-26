#Importing all the necessary libraries and packages
import os 
import streamlit as st 
import pandas as pd
from streamlit_option_menu import option_menu
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
from pycaret.classification import ClassificationExperiment
# from sklearn.datasets import load_diabetes
# import lazypredict
# from lazypredict.Supervised import LazyClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.utils import all_estimators #for lazypred
# from sklearn.base import ClassifierMixin #for lazypred
# import matplotlib.pyplot as plt #for lazypred
# import seaborn as sns #for lazypred
# import base64 #for lazypred
# import io #for lazypred
# import time #tpot
from streamlit_shap import st_shap
import shap
import numpy as np
# from AutoClean import AutoClean
import AutoDataCleaner.AutoDataCleaner as adc

#streamlit-wide-layout
st.set_page_config(layout="wide",page_title="AutoML by Nikhita") 

# To hide the default header and footer of the streamlit app
# hide_default_format = """
#        <style>
#        #MainMenu {visibility: hidden; }
#        footer {visibility: hidden;}
#        </style>
#        """
# st.markdown(hide_default_format, unsafe_allow_html=True)

# Load the sample file from local folder
if os.path.exists('dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

# Set default session state variables
st.session_state.setdefault('df', None)
st.session_state.setdefault('tuned_model', None)
st.session_state.setdefault('best', None)
st.session_state.setdefault('chosen_target', None)

# Function to Download CSV data 
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
# def filedownload(df, filename):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
#     href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
#     return href

# Function to download image files of charts
# def imagedownload(plt, filename):
#     s = io.BytesIO()
#     plt.savefig(s, format='pdf', bbox_inches='tight')
#     plt.close()
#     b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
#     href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
#     return href

# title of the web application
st.title("Welcome to iBioML - AutoML")

# adding a text description
st.text("This web application allows users to automate their machine learning processes by just uploading their data. ")

# # adding a horizontal divider
# st.divider()

#adding a horizontal menu
selected = option_menu(
    menu_title=None,
    options=["Home","Data Information", "EDA","Data Cleaning","AutoML","Interpretability"],
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
        # count_row = df.shape[0]
        # #Temp solution for heavy data, considering a subset
        # if count_row > 100000:
        #     df = df.sample(50000)
        st.dataframe(df) 
        st.text("To Auto-Analyze your data click the next Tab - Data Information!")
        if st.button("Next Tab"):
            selected="Data Information"
    elif(st.button("[Diabetes_dataset.csv](http://nik-automl.streamlit.app/app/static/diabetes_dataset.csv)")):
        if pd.read_csv("http://localhost:8501/app/static/diabetes_dataset.csv") is not None:
            df = pd.read_csv("http://localhost:8501/app/static/diabetes_dataset.csv")
            df.to_csv('dataset.csv', index=None)
        # else:
        #     diabetes = load_diabetes()
        #     X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        #     Y = pd.Series(diabetes.target, name='response')
        #     df = pd.concat( [X,Y], axis=1 )
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
        

#----------------- EDA (Data analysis)--------------------------------------------------------
if selected == "EDA":
    st.header("Auto-analysis - EDA")
    profile = ProfileReport(df, title="Profiling Report",explorative=True, dark_mode=True)
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
        
        
#----------------- Data Cleaning (Data Preprocessing)--------------------------------------------------------
if selected == "Data Cleaning":
    st.header("Data Cleaning")
    
    if df is not None: 
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
            # pipeline = AutoClean(data, mode=mode, duplicates=duplicates, missing_num=missing_num, missing_categ=missing_categ, 
            # encode_categ=encode_categ, extract_datetime=False, outliers=outliers, outlier_param=1.5, 
            # logfile=True, verbose=verbose)
            # st.write(pipeline.output)
            # df = pipeline.output
            new_data = adc.clean_me(data, detect_binary = detect_binary, one_hot = one_hot, na_cleaner_mode=missing_num, normalize=normalize, remove_columns=remove_columns, verbose=verbose)
            st.write(new_data)
    else:
        st.__loader__
 # else:
 #     st.error("Oops looks like your data is unavailable. Please try uploading you data again.")


#----------------- PyCaret (AutoML using PyCaret)--------------------------------------------------------        
if selected == "AutoML":
    print("This works")
    st.header("AutoML using PyCaret")
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    st.session_state['chosen_target'] = chosen_target
    if chosen_target and chosen_target != df.columns[0]:
        s = ClassificationExperiment()
        s.setup(df, target=chosen_target,normalize=True, 
            transformation=True)
        setup_df = s.pull()
        st.dataframe(setup_df)
        st.write("Data after cleaning and transformation:")
        st.write(s.dataset_transformed)
        # st.header("Run AutoML")
        # df[(df['diabetes'] == 1) | (df['diabetes'] == 0)]
        # best_model = compare_models()
        best = s.compare_models() 
        # progress_text = "Operation in progress. Please wait."
        # my_bar = st.progress(0, text=progress_text)
        
        if best:
            compare_df = s.pull()
            st.dataframe(compare_df)
            # bmodel = list(compare_df)
            st.header("Best Model")
            st.text(best)
            # st.dataframe(best_model)
            # evaluate_model(compare_df)
            s.evaluate_model(best)
            # model = s.create_model(compare_df[''][0], feature_selection=True, feature_interaction=True, feature_ratio=True)
            st.write(best)
            st.session_state['best'] = best
            temp_df = s.dataset_transformed
            st.session_state['df'] = temp_df
            # Tune the best model
            tuned_model = s.tune_model(best)
            st.write(tuned_model)
            st.session_state['tuned_model'] = tuned_model
            # # Explain the model using SHAP values
            # if(TypeError(shap.Explainer(tuned_model))):
            #     explainer = shap.TreeExplainer(best)
            #     shap_values = explainer.shap_values(temp_df.drop(chosen_target, axis=1))
            # else:
            #     explainer = shap.Explainer(tuned_model)
            #     shap_values = explainer.shap_values(temp_df.drop(chosen_target, axis=1))

            # st.write(np.shape(shap_values))

            # st.write(s.interpret_model(best, plot='summary'))
            # Display additional information or visualizations based on the model output
            # Use interpret_model to generate and display charts
            # st.subheader("Visualizations")
            # interpret_model(best_model)

            # Show Streamlit charts based on the evaluation
            st.subheader("Model Evaluation Charts")
  
            s.plot_model(best, plot = 'confusion_matrix', display_format='streamlit')
            s.plot_model(best, plot = 'auc', display_format='streamlit')
            s.plot_model(best, plot = 'feature', display_format='streamlit') 

            s.plot_model(best, plot = 'feature_all', display_format='streamlit') 

            # interactive(children=(ToggleButtons(description='Plot Type:', icons=('',), options=(('Pipeline Plot', 'pipeline'), ('Hyperparameters', 'parameter'), ('AUC', 'auc'), ('Confusion Matrix', 'confusion_matrix'), ('Threshold', 'threshold'), ('Precision Recall', 'pr'), ('Prediction Error', 'error'), ('Class Report', 'class_report'), ('Feature Selection', 'rfe'), ('Learning Curve', 'learning'), ('Manifold Learning', 'manifold'), ('Calibration Curve', 'calibration'), ('Validation Curve', 'vc'), ('Dimensions', 'dimension'), ('Feature Importance', 'feature'), ('Feature Importance (All)', 'feature_all'), ('Decision Boundary', 'boundary'), ('Lift Chart', 'lift'), ('Gain Chart', 'gain'), ('Decision Tree', 'tree'), ('KS Statistic Plot', 'ks')), value='pipeline'), Output()), _dom_classes=('widget-interact',))
            # st.subheader("SHAP Summary Plot:")
            # st_shap(shap.summary_plot(shap_values, temp_df.drop(chosen_target, axis=1))) 
            
            # st_shap(shap.plots.waterfall(shap_values[0]))
            # st_shap(shap.plots.beeswarm(shap_values))

            # st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:]))
            # st_shap(shap.force_plot(explainer.expected_value, shap_values[:1000,:]))

        else:
            st.__loader__

            
#--------------------------SHAP (Interpretability)-------------------------------------
if selected == "Interpretability":
    st.header("Interpretability using SHAP")  
    st.subheader("Oops! looks like something isn't working right, we are working on it. Thank you for your patience.")  
    tuned_model = st.session_state['tuned_model']
    temp_df = st.session_state['df']
    chosen_target = st.session_state['chosen_target']
    best = st.session_state['best']
    # Explain the model using SHAP values
    if(TypeError(shap.Explainer(tuned_model))):
        explainer = shap.TreeExplainer(best)
        shap_values = explainer.shap_values(temp_df.drop(chosen_target, axis=1))
    else:
        explainer = shap.Explainer(tuned_model)
        shap_values = explainer.shap_values(temp_df.drop(chosen_target, axis=1))

    st.write(np.shape(shap_values))

    st.subheader("SHAP Summary Plot:")
    st_shap(shap.summary_plot(shap_values, temp_df.drop(chosen_target, axis=1))) 
    
    # st_shap(shap.plots.waterfall(shap_values[0]))
    # st_shap(shap.plots.beeswarm(shap_values),height=300,width=200)
    st_shap(shap.plots.bar(shap_values),height=300,width=200)

    st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:]))
    st_shap(shap.force_plot(explainer.expected_value, shap_values[:1000,:]))
