import streamlit as st
from streamlit_option_menu import option_menu

# Define the pages and their file paths
pages = {'Home':'Home.py',
         'Data Information':'pages/information.py',
         'Data Analysis':'pages/eda.py',
         'Data Cleaning':'pages/cleaning.py',
         'AutoML':'pages/automl.py',
         'Interpretability':'pages/interpretability.py'}

# Create a list of the page names
page_list = list(pages.keys())

# Method to navigate pages
def nav(current_page=page_list[0]):

    p = option_menu(None, page_list, 
        menu_icon="cast",
        default_index=page_list.index(current_page), 
        orientation="horizontal")

    if current_page != p:
        st.switch_page(pages[p])