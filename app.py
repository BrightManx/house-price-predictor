import streamlit as st
from backend import show_explore, show_try

st.set_page_config(
    page_title="House Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

hide_menu_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Sidebar 
pages = ['Try the model', 'Explore the model']
page = st.sidebar.selectbox('', pages)

# Pages
if page == 'Try the model':
    show_try()

if page == 'Explore the model':
    show_explore()
