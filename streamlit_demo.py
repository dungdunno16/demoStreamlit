import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from datetime import time,datetime

st.header('Demo')

if st.button('Say hello'):
    st.write('Why hello there')
else:
    st.write('Goodbye')

st.write('Hello! *World!* :sunglasses:')

st.write(1234)
df = pd.DataFrame({
    'first column': [1,2,3,4],
    'second column': [10,20,30,40]
})
st.write(df)

st.write('Below is a DataFrame', df, 'Above is a DataFrame')

df2 = pd.DataFrame(
    np.random.rand(200,3),
    columns=['a','b','c']
)
c = alt.Chart(df2).mark_circle().encode(
    x = 'a', y = 'b', size = 'c',color = 'c', tooltip=['a','b','c']
)
st.write(c)

age = st.slider('Age',0,100,22)
st.write("I'm", age, "years old")

values = st.slider('Values',0.0,100.0,(25.0,70.0))
st.write("Values: ", values)

appointment = st.slider('Scheduled appointment',value=(time(11,30),time(12,45)))
st.write("Appointment: ", appointment)

start_time = st.slider('Start time: ', value=datetime(2025,1,1,9,30), format="MM/DD/YYYY hh:mm")
st.write("Start time: ",start_time)