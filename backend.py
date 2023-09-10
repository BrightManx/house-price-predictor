import streamlit as st
import pandas as pd
import numpy as np
import joblib
from preprocessing import preprocess

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

def show_try():

    enc = joblib.load('enc.joblib')
    scaler = joblib.load('scaler.joblib')
    forest = joblib.load('forest.joblib')

    with st.form('form', clear_on_submit=False):

        st.write("<h1 style='text-align:center'> How Much is Your House? </h1>",unsafe_allow_html=True)
        st.write("<p style='text-align:center'> Tell us some features about your house to predict its price. </p>",unsafe_allow_html=True)
        st.write("<p style='text-align:center'> Remember: the more features you add, the more accurate the prediction will be. </p>",unsafe_allow_html=True)
        st.write("<p style='text-align:center'>[Note: empty fields will be guessed by the model] </p>",unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.selectbox('conditions', key='conditions', options=enc.categories_[0] )
            st.number_input('construction_year', key='construction_year', step = 1, value = 2000)
            st.number_input('latitude', key='latitude', value = np.nan)
            st.number_input('longitude', key='longitude', value = np.nan)
            st.number_input('energy_efficiency', key='energy_efficiency', value = np.nan)
            st.number_input('expenses', key='expenses', value = np.nan)
            st.number_input('floor', key='floor', value = np.nan, step = 1.0)

        with col2:
            st.number_input('n_bathrooms', key='n_bathrooms', value = np.nan, step = 1.0)
            st.number_input('total_floors', key='total_floors', value = np.nan, step = 1.0)
            st.number_input('n_rooms', key='n_rooms', value = np.nan, step = 1.0)
            st.number_input('proximity_to_center', key='proximity_to_center', value = np.nan, step = 1.0)
            st.number_input('surface (m^2)', key='surface', value = np.nan, step = 1.0)
            st.checkbox('balcony', key='balcony')
            st.checkbox('garden', key='garden')
            st.checkbox('elevator', key='elevator')
            submit = st.form_submit_button()

    if submit:
        
        # Store userHouse in session_state
        if 'userHouse' in st.session_state.keys():
            st.session_state.pop('userHouse')
        st.session_state['userHouse'] = pd.DataFrame(dict(st.session_state), index = [0]).drop(columns="FormSubmitter:form-Submit")
        
        # Pre-process the userHouse
        X = preprocess(st.session_state['userHouse'])
        X = X.to_numpy()
        X = scaler.transform(X)

        # Predict the userHouse price
        pred = forest.predict(X)
        st.write("<p style='text-align:center'> Your house is predicted to be worth </p>",unsafe_allow_html=True)
        st.write(f"<h1 style='text-align:center'> € {pred.item():,.2f} </h1>",unsafe_allow_html=True)

    return


def show_explore():

    st.write("<h1 style='text-align:center;'> Explore the model </h1>", unsafe_allow_html=True)
    st.write("<p style='text-align:center'> work in progress... </p>",unsafe_allow_html=True)

    return
