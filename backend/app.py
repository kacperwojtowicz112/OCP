import streamlit as st
import requests
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Aplikacja Streamlit')
with st.sidebar:
    selected_option = option_menu("Main Menu", ["See the quality of the model", "Imput your data"], default_index=0)
brands=['All','hyundi', 'volkswagen', 'BMW', 'skoda', 'ford', 'toyota', 'merc',
       'vauxhall', 'Audi']
user_input = st.selectbox("Select brand:", brands)
if st.button('Przekazuj dane'):
    st.write(user_input)
    response = requests.post('http://localhost:8501/process', json={'data': user_input})
    st.write(response.text)
    response = requests.get('http://localhost:8501/app')
    data = response.json()

    serialized_model = data.get('model', None)
    if serialized_model:
        model = pickle.loads(serialized_model)
        confusion_matrix_data = data.get('confusion_matrix', None)
        mse=data.get('mse',None)

    if selected_option=='See the quality of the model':
        st.write("MSE:",mse)
        cm = np.array(confusion_matrix_data)
        sns.heatmap(cm, annot=True, fmt="d")