import streamlit as st
import pandas as pd
import numpy as np
import pickle

file = open('pipe.pkl', 'rb')
rf = pickle.load(file)
file.close()

data = pd.read_csv('traineddata.csv')

data['IPSPanel'].unique()

st.title('Laptop Price Predictor')

company = st.selectbox('Brand', data['Company'].unique())

type = st.selectbox('Type', data['TypeName'].unique())

ram = st.selectbox('Ram', data['Ram'].unique())

os = st.selectbox('OS', data['OpSys'].unique())

screen_size = st.number_input('Screen Size', min_value=1)

weight = st.number_input('Weight of the laptop')

touchscreen = st.number_input(
    'Touch Screen (1 for Yes 0 for No)', min_value=0, max_value=1)

ips = st.number_input('IPS Panel (1 for Yes 0 for No)',
                      min_value=0, max_value=1)

resolution = st.selectbox('Screen Resolution', [
                          '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1440'])

cpu = st.selectbox('CPU', data['Cpu_Name'].unique())

gpu = st.selectbox('GPU', data['Gpu_Brand'].unique())

submit = st.button('Predict Price')


if submit:
    ppi = None


    x_resolution = int(resolution.split('x')[0])
    y_resolution = int(resolution.split('x')[1])

    ppi = ((x_resolution**2)+(y_resolution**2))**0.5/(screen_size)

    query = np.array([company, type, ram, os, weight,
                    touchscreen, ips, ppi, cpu, gpu])

    query = query.reshape(1, 10)

    prediction = int(np.exp(rf.predict(query)[0]))

    st.title(
        f'Predicted Price for this laptop could be between ₹{str(prediction-1000)} to ₹{str(prediction+1000)}')
