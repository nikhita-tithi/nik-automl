#Importing all the necessary libraries and packages
import streamlit as st 
import navbar
from pycaret.classification import ClassificationExperiment, plot_model

#streamlit-wide-layout
st.set_page_config(layout="wide",page_title="AutoML by Nikhita") 

# Setting the navigation bar
current_page = "AutoML"
st.header(current_page)

navbar.nav(current_page)

def run_automl(data):
    chosen_target = st.selectbox('Choose the Target Column', data.columns)
    st.session_state['chosen_target'] = chosen_target
    if chosen_target and chosen_target != data.columns[0]:
        s = ClassificationExperiment()
        s.setup(data, target=chosen_target,normalize=True, 
            transformation=True)
        setup_data = s.pull()
        st.dataframe(setup_data)
        st.write("Data after cleaning and transformation:")
        st.write(s.dataset_transformed)
        # st.header("Run AutoML")
        # data[(data['diabetes'] == 1) | (data['diabetes'] == 0)]
        # best_model = compare_models()
        best = s.compare_models() 
        # progress_text = "Operation in progress. Please wait."
        # my_bar = st.progress(0, text=progress_text)
        
        if best:
            compare_data = s.pull()
            st.dataframe(compare_data)
            # bmodel = list(compare_data)
            st.header("Best Model")
            st.text(best)
            # st.dataframe(best_model)
            # evaluate_model(compare_data)
            s.evaluate_model(best)
            # model = s.create_model(compare_data[''][0], feature_selection=True, feature_interaction=True, feature_ratio=True)
            st.write(best)
            st.session_state['best'] = best
            temp_data = s.dataset_transformed
            st.session_state['df'] = temp_data
            # Tune the best model
            tuned_model = s.tune_model(best)
            # st.write(tuned_model)
            st.session_state['tuned_model'] = tuned_model
            # # Explain the model using SHAP values
            # if(TypeError(shap.Explainer(tuned_model))):
            #     explainer = shap.TreeExplainer(best)
            #     shap_values = explainer.shap_values(temp_data.drop(chosen_target, axis=1))
            # else:
            #     explainer = shap.Explainer(tuned_model)
            #     shap_values = explainer.shap_values(temp_data.drop(chosen_target, axis=1))

            # st.write(np.shape(shap_values))

            # st.write(s.interpret_model(best, plot='summary'))
            # Display additional information or visualizations based on the model output
            # Use interpret_model to generate and display charts
            # st.subheader("Visualizations")
            # s.interpret_model(best)

            # Show Streamlit charts based on the evaluation
            st.subheader("Model Evaluation Charts")
            # s.plot_model(best,plot='diagnostics',display_format='streamlit')
            # st.write(s.check_fairness(best))
            s.plot_model(best, plot = 'confusion_matrix', display_format='streamlit')
            s.plot_model(best, plot = 'auc', display_format='streamlit')
            s.plot_model(best, plot = 'feature', display_format='streamlit')

            s.plot_model(best, plot = 'feature_all', display_format='streamlit')

            # interactive(children=(ToggleButtons(description='Plot Type:', icons=('',), options=(('Pipeline Plot', 'pipeline'), ('Hyperparameters', 'parameter'), ('AUC', 'auc'), ('Confusion Matrix', 'confusion_matrix'), ('Threshold', 'threshold'), ('Precision Recall', 'pr'), ('Prediction Error', 'error'), ('Class Report', 'class_report'), ('Feature Selection', 'rfe'), ('Learning Curve', 'learning'), ('Manifold Learning', 'manifold'), ('Calibration Curve', 'calibration'), ('Validation Curve', 'vc'), ('Dimensions', 'dimension'), ('Feature Importance', 'feature'), ('Feature Importance (All)', 'feature_all'), ('Decision Boundary', 'boundary'), ('Lift Chart', 'lift'), ('Gain Chart', 'gain'), ('Decision Tree', 'tree'), ('KS Statistic Plot', 'ks')), value='pipeline'), Output()), _dom_classes=('widget-interact',))
            # st.subheader("SHAP Summary Plot:")
            # st_shap(shap.summary_plot(shap_values, temp_data.drop(chosen_target, axis=1))) 
            
            # st_shap(shap.plots.waterfall(shap_values[0]))
            # st_shap(shap.plots.beeswarm(shap_values))

            # st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:]))
            # st_shap(shap.force_plot(explainer.expected_value, shap_values[:1000,:]))

            # Going to the next tab
            st.markdown("To run interpretability on your model click on the button below!")
            st.page_link('pages/interpretability.py', label="Next Tab - Interpretability")

#----------------- PyCaret (AutoML using PyCaret)--------------------------------------------------------        

st.header("AutoML using PyCaret")

if 'df' in st.session_state and st.session_state['df'] is not None: 

    # Fetching session state variables
    df = st.session_state['df']
    cleaned_data = st.session_state['cleaned_data']

    if df is not None and cleaned_data is not None:
        data_for_ml = st.selectbox('Which data would you like to use - Original data or Cleaned data?',['Original data','Cleaned data'])
        if data_for_ml == 'Original data':
            run_automl(df)
        elif data_for_ml == 'Cleaned data':
            run_automl(cleaned_data)
    elif df is not None:
        data_for_ml = df
        run_automl(df)
else:
     st.error("Oops looks like your data is unavailable. Please try uploading you data again.")

    # Going to the previous tab
     st.markdown("To go back and re-upload your data click on the button below!")
     st.page_link('Home.py', label="Back - Re-upload Data")