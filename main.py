import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import plotly.express as px
import pickle
from sklearn.preprocessing import StandardScaler
import warnings


class RegressionNN(nn.Module):
    def __init__(self):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(9, 64)  # Input layer
        self.fc2 = nn.Linear(64, 64)  
        self.fc3 = nn.Linear(64, 1) 

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  
        return x
    

warnings.simplefilter("ignore")

st.sidebar.title("DA SPNP Prediction Application")
csv = st.sidebar.file_uploader("Upload CSV data for DA SPNP", type="csv")

if csv is not None:
    df = pd.read_csv(csv)
    if st.checkbox('Show raw data'):
        st.write("Uploaded Data:")
        st.dataframe(df.head())

    df = df.dropna()
    try:
        model = RegressionNN()
        random_forest_model = pickle.load(open("./Saved Models/random_forest_model_v2.pkl", 'rb'))

        model = RegressionNN()
        model.load_state_dict(torch.load("./Saved Models/regression_nn_model_v2.pth"))
        model.eval()

        df.columns = df.columns.str.strip()
        dates = pd.to_datetime(df['Date/Time'])
        df = df.rename(columns={df.columns[-1]: "DA SPNP"})

        try:
            df['NP15_LOAD'] = df['NP15_LOAD'].str.replace(',', '').astype('float64')
            df['SP15_LOAD'] = df['SP15_LOAD'].str.replace(',', '').astype('float64')

            # Calculate deltas and sum
            df['PGE_Malin_Delta'] = df['PG&E'] - df['Malin']
            df['Load_Delta'] = df['NP15_LOAD'] - df['SP15_LOAD']
            df['Load_Sum'] = df['NP15_LOAD'] + df['SP15_LOAD']

            # Drop specified columns
            drop_columns = ['DA SPNP', 'Date/Time', "NP15_LOAD", "Malin", "PG&E", 'SP15_LOAD']
            df = df.drop(columns=drop_columns, errors='ignore')
        except Exception as e:
            pass

        df = df.drop(columns=['SP15 (SOLAR_FORECAST Latest) - NP15 (SOLAR_FORECAST Latest) Maximum', 'Date/Time', 'DA SPNP'], errors='ignore')

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '')  # Remove commas
                df[col] = df[col].str.strip()  # Strip whitespace
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric
        
        scaler = StandardScaler()
        df_scaled_array = scaler.fit_transform(df)

        # Convert the scaled array back to a DataFrame with original column names
        df_scaled = pd.DataFrame(df_scaled_array, columns=df.columns)

        if st.checkbox('Show processed data'):
            st.write("Scaled Training Data:")
            st.dataframe(df_scaled)

        with torch.no_grad():
            y_pred = model(torch.tensor(df_scaled.values, dtype=torch.float32))

        y_pred = y_pred.flatten().numpy()
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
