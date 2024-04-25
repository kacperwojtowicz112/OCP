import streamlit as st
import requests
import pandas as pd
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Welcome")
def train_model(brand):
    response_preprocess = requests.post('http://127.0.0.1:8000/train/', json = brand)
    rmse = response_preprocess.json()["rmse"]
    mae = response_preprocess.json()["mae"]
    return rmse, mae
with st.sidebar:
    selected_option = option_menu("Main Menu", ["Choose an option","Train model", "Make prediction"], default_index=0)

if selected_option=="Choose an option":
    st.write("Choose an option")
elif selected_option=="Train model":
    st.header("Model training")
    col1,col2 = st.columns(2)
    with st.form(key = "train"):
            brand = st.selectbox("Choose your brand",['hyundi', 'volkswagen', 'BMW', 'skoda', 'ford', 'toyota', 'merc','vauxhall', 'Audi'],
                placeholder="Choose an option",
                index = None)
            brand = {'brand': brand}
            rmse, mae= train_model(brand)
            submit_button = st.form_submit_button("Click here to confirm!", type="primary")
            st.write("RMSE:", rmse)
            st.write("MAE:", mae)
        
else:
    def predict_data(file_bytes):
        response_predict = requests.post('http://127.0.0.1:8000/predict/', files={'file': file_bytes})
        json_data = response_predict.json()["pred"]
        if json_data!="":
            df = pd.DataFrame(json_data)
        else:
            df = ""
        return df

    data = ""

    input_file = st.file_uploader("Upload a CSV file", type="csv", accept_multiple_files=False)

    if input_file is not None:
        st.info("Success")
        if st.button("Upload"):
            st.info("Uploading...")
            file_bytes = input_file.getvalue()
            data = predict_data(file_bytes)
            st.dataframe(data, width=150)
            file = pd.convert_df(data)
            st.download_button(
            label="Download predictions as CSV",
            data=file,
            file_name='predictions.csv',
            mime='text/csv',)

