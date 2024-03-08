import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import ClassificationExperiment
import os 
# from sklearn.datasets import load_diabetes
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

# Importing interpret_model from pycaret.classification
# from pycaret.classification import interpret_model

# # Importing get_config from pycaret.internal.Display
# from pycaret.internal import get_config

if os.path.exists('dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

# Download CSV data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

# title
st.title("Welcome to iBioML - AutoML")
st.divider()

# adding a horizontal menu
selected = option_menu(
    menu_title=None,
    options=["Home","EDA","PyCaret","LazyPredict","TPOT"],
    icons=["house"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    )

#----------------- HOME (upload or select data)--------------------------------------------------------
if selected == "Home":
    st.header("Upload Your Dataset")
    file = st.file_uploader("") #leaving it empty as we have already given a header with a label
    st.text("Or you can use this sample dataset:")
    # st.button("[Diabetes_dataset.csv](http://nik-automl.streamlit.app/app/static/diabetes_dataset.csv)")
    # with st.echo():
    #     st.file_uploader(
    #         <a href="./app/static/diabetes_dataset.csv">,
    #         unsafe_allow_html=True,
    #     )
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df) 
        st.text("To Auto-Analyze your data click next!")
        if st.button("Next"):
            selected="EDA"
    elif(st.button("[Diabetes_dataset.csv](http://nik-automl.streamlit.app/app/static/diabetes_dataset.csv)")):
        df = pd.read_csv("http://localhost:8501/app/static/diabetes_dataset.csv")
        df.to_csv('dataset.csv', index=None)
        # diabetes = load_diabetes()
        # X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        # Y = pd.Series(diabetes.target, name='response')
        # df = pd.concat( [X,Y], axis=1 )
        st.dataframe(df) 
        st.text("To Auto-Analyze your data click next!")
        if st.button("Next"):
            selected="EDA"
    else:
        st.error("Please Upload a .csv file!")


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


#----------------- PyCaret (AutoML using PyCaret)--------------------------------------------------------        
if selected == "PyCaret":
    st.header("AutoML using PyCaret")
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if chosen_target and chosen_target != df.columns[0]:
        s = ClassificationExperiment()
        s.setup(df, target=chosen_target,normalize=True, 
            transformation=True)
        setup_df = s.pull()
        st.dataframe(setup_df)
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

            # tuned_model = s.tune_model(best, n_iter=5, search_library='optuna')
            # st.write(tuned_model)

            # st.write(s.interpret_model(best, plot='summary'))
            # Display additional information or visualizations based on the model output
            # Use interpret_model to generate and display charts
            # st.subheader("Visualizations")
            # interpret_model(best_model)

            # Show the charts generated by interpret_model
            # plt = get_config("plt")
            # charts = plt.show()
            # st.pyplot(charts)

            # Show Streamlit charts based on the evaluation
            st.subheader("Model Evaluation Charts")

            # Loop through metrics and generate visualizations for the best model
            # for metric in ['auc', 'accuracy']:
            #     s.plot_model(best_model_based_on_auc, plot=metric)
            #     st.pyplot()
  
            s.plot_model(best, plot = 'confusion_matrix', display_format='streamlit')
            s.plot_model(best, plot = 'auc', display_format='streamlit')
            # s.plot_model(best, plot = 'feature', display_format='streamlit') 

            st.subheader("Model Evaluation Charts")

            s.plot_model(best, plot = 'feature_all', display_format='streamlit') 

            # interactive(children=(ToggleButtons(description='Plot Type:', icons=('',), options=(('Pipeline Plot', 'pipeline'), ('Hyperparameters', 'parameter'), ('AUC', 'auc'), ('Confusion Matrix', 'confusion_matrix'), ('Threshold', 'threshold'), ('Precision Recall', 'pr'), ('Prediction Error', 'error'), ('Class Report', 'class_report'), ('Feature Selection', 'rfe'), ('Learning Curve', 'learning'), ('Manifold Learning', 'manifold'), ('Calibration Curve', 'calibration'), ('Validation Curve', 'vc'), ('Dimensions', 'dimension'), ('Feature Importance', 'feature'), ('Feature Importance (All)', 'feature_all'), ('Decision Boundary', 'boundary'), ('Lift Chart', 'lift'), ('Gain Chart', 'gain'), ('Decision Tree', 'tree'), ('KS Statistic Plot', 'ks')), value='pipeline'), Output()), _dom_classes=('widget-interact',))
        else:
            st.__loader__
            # for percent_complete in range(100):
            #     time.sleep(0.1)
            #     my_bar.progress(percent_complete + 1, text=progress_text)
        # save_model(best_model, 'best_model') 
    # else:
    #     st.__loader__
            

#----------------- LazyPredict (AutoML using LazyPredict)--------------------------------------------------------
if selected == "LazyPredict":   
    st.header("AutoML using LazyPredict")
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    removed_classifiers = [
    "ClassifierChain",
    "ComplementNB",
    "GradientBoostingClassifier",
    "GaussianProcessClassifier",
    "HistGradientBoostingClassifier",
    "MLPClassifier",
    "LogisticRegressionCV",
    "MultiOutputClassifier",
    "MultinomialNB",
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
    "RadiusNeighborsClassifier",
    "VotingClassifier",
    'SVC','LabelPropagation','LabelSpreading','NuSV']
    classifiers_list = [est for est in all_estimators() if (issubclass(est[1], ClassifierMixin) and (est[0] not in removed_classifiers))]
    if chosen_target and chosen_target != df.columns[0]:
        data = df.copy()
        y= data[chosen_target]
        data.drop(chosen_target,axis =1)
        X = data.drop(chosen_target,axis =1)

        # Display X and y values
        st.markdown('Dataset dimensions')
        st.text("X values:")
        st.write(X)
        st.info(X.shape)
        st.text("Y values:")
        st.write(y)
        st.info(y.shape)

        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.20,random_state =123)
        clf = LazyClassifier(verbose=1,ignore_warnings=False, custom_metric=None, classifiers = classifiers_list)
        models,predictions = clf.fit(X_train, X_test, y_train, y_test)
        if models is not None:
            st.text("Classification models:")
            # Display the models
            st.write(models)

            st.subheader('Plot of Model Performance:')
            chart_type = st.selectbox('Chart Alignment', ['Wide','Tall'])
            type = st.selectbox('Choose the Performance parameter', predictions.keys())

            st.markdown(type)
                # Tall
            if chart_type == 'Tall':
                predictions[type] = [0 if i < 0 else i for i in predictions[type] ]
                plt.figure(figsize=(5, 9))
                sns.set_theme(style="whitegrid")
                ax1 = sns.barplot(y=predictions.index, x=type, data=predictions)
                ax1.set(xlim=(0, 1))
                st.pyplot(plt)
                st.markdown(imagedownload(plt,'plot-r2-tall.pdf'), unsafe_allow_html=True)
                
            # Wide
            else:
                plt.figure(figsize=(9, 5))
                sns.set_theme(style="whitegrid")
                ax1 = sns.barplot(x=predictions.index, y=type, data=predictions)
                ax1.set(ylim=(0, 1))
                plt.xticks(rotation=90)
                st.pyplot(plt)
                st.markdown(imagedownload(plt,'plot-r2-wide.pdf'), unsafe_allow_html=True)
            # compute SHAP values
                
            # # Extract the best model
            # mdl = models.iloc[0]

            # # Train the best model on the full dataset
            # mdl.fit(X, y)

            # # Use SHAP to explain the model predictions
            # explainer = shap.Explainer(mdl)
            # shap_values = explainer.shap_values(X)

            # # Visualize SHAP summary plot
            # shap.summary_plot(shap_values, X, feature_names=df.feature_names, class_names=df.target_names)
        else:
            st.__loader__

