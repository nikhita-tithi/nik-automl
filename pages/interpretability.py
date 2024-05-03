#Importing all the necessary libraries and packages
import streamlit as st 
import navbar
from streamlit_shap import st_shap
import shap
import numpy as np

#streamlit-wide-layout
st.set_page_config(layout="wide",page_title="AutoML by Nikhita") 

# Setting the navigation bar
current_page = "Interpretability"
st.header(current_page)

navbar.nav(current_page)

#--------------------------SHAP (Interpretability)-------------------------------------

st.header("Interpretability using SHAP")

# Fetching session state variables
if 'df' not in st.session_state:
    st.error("Oops looks like your data is unavailable. Please try uploading you data again.")

    # Going to the previous tab
    st.markdown("To go back and re-upload your data click on the button below!")
    st.page_link('Home.py', label="Back - Re-upload Data")
else:

    df = st.session_state['df']
    cleaned_data = st.session_state['cleaned_data']
    tuned_model = st.session_state['tuned_model']
    temp_df = st.session_state['df']
    chosen_target = st.session_state['chosen_target']
    best = st.session_state['best']

    # Explain the model using SHAP values
    if tuned_model is not None and best is not None:
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
    else:
        st.subheader('''Oops! looks like something went wrong, 
                    
                    Please ensure that you run AutoML completely and successfully before using interpretability!!''')  