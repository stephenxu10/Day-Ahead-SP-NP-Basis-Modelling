import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import os
from sklearn.preprocessing import StandardScaler

# Improved layout
st.sidebar.title("DA SPNP Prediction Application")
csv = st.sidebar.file_uploader("Upload CSV data for DA SPNP", type="csv")

if csv is not None:
    df = pd.read_csv(csv)
    if st.checkbox('Show raw data'):
        st.write("Uploaded Data:")
        st.dataframe(df.head())

    df = df.dropna()
    try:
        random_forest_model = pickle.load(open("./Saved Models/random_forest_model.pkl", 'rb'))
        
        df.columns = df.columns.str.strip()
        dates = pd.to_datetime(df['Date/Time'])

        drop_columns = ['DA SPNP', 'Date/Time', "NP15_LOAD", "Malin", "SOCAL"]
        df = df.drop(columns=drop_columns, errors='ignore')

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '')  # Remove commas
                df[col] = df[col].str.strip()  # Strip whitespace
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric
        
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
        
        if st.checkbox('Show scaled data'):
            st.write("Scaled Training Data:")
            st.dataframe(df_scaled)

        y_pred = random_forest_model.predict(df_scaled)
        
        df_pred = pd.DataFrame({'Date/Time': dates, 'DA SPNP': y_pred})
        if st.checkbox('Show predictions'):
            st.write("Predictions:")
            st.dataframe(df_pred)

        pred_csv = df_pred.to_csv(index=False)
        st.download_button('Download Predictions', pred_csv, 'predictions.csv', 'text/csv')

        # Use Plotly for interactive plotting
        fig = px.line(df_pred, x='Date/Time', y='DA SPNP', title='Prediction Over Time')
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.write("Awaiting CSV file to be uploaded. Currently, no file is uploaded.")
