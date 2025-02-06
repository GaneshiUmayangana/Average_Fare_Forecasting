import streamlit as st
import pandas as pd
from datetime import timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from itertools import product
import numpy as np

# Streamlit page configuration
st.set_page_config(
    page_title="Forecasting Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)
st.header("Average Yield Prediction")

# File upload section
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    # Load the uploaded file into a DataFrame
    df = pd.read_excel(uploaded_file)

    # Convert 'Sale Date' and 'Flight Date' to datetime
    df["Sale Date"] = pd.to_datetime(df["Sale Date"])
    df["Flight Date"] = pd.to_datetime(df["Flight Date"])

    # Section 1: Select forecast period
    forecast_window_start = df["Sale Date"].min()
    forecast_window_end = df["Sale Date"].max()

    # Select forecast start date
    forecast_period_start = st.date_input(
        "Select Forecast Start Date",
        min_value=forecast_window_start,
        max_value=forecast_window_end
    )

    # Select forecast end date
    forecast_period_end = st.date_input(
        "Select Forecast End Date",
        min_value=forecast_period_start,  # Ensure the end date is after or equal to the start date
        max_value=forecast_window_end     # Ensure the end date is before or equal to the departure date
    )

    # Validate the selected range and display it
    if forecast_period_start and forecast_period_end:
        st.write(f"Selected Forecast Period: {forecast_period_start} to {forecast_period_end}")

    # Add a button to generate the forecasts
    if st.button("Generate Forecast"):
        # Filter the data based on the forecast period selected
        df_filtered = df[(df["Sale Date"] >= pd.to_datetime(forecast_period_start)) & 
                         (df["Sale Date"] <= pd.to_datetime(forecast_period_end))]

        # Group the filtered data by 'Sector' and calculate the average yield
        df_grouped = df_filtered.groupby("Sector").agg(
            Avg_YLD_USD=("YLD USD", "mean")
        ).reset_index()

        # Prepare the data for Exponential Smoothing
        df_forecast_data = df_filtered.groupby("Sale Date", as_index=False).agg(
            Avg_YLD_USD=("YLD USD", "mean")
        )

        # Train Exponential Smoothing model on the data up to forecast_period_start
        y_train = df_forecast_data["Avg_YLD_USD"]

        # Hyperparameter tuning for Exponential Smoothing
        seasonal_periods_list = [7, 30]  # Weekly and monthly seasonality
        trend_types = ['add', 'mul']
        seasonal_types = ['add', 'mul']

        best_model = None
        best_rmse = float('inf')

        for trend, seasonal, seasonal_periods in product(trend_types, seasonal_types, seasonal_periods_list):
            try:
                exp_smooth_model = ExponentialSmoothing(
                    y_train, 
                    trend=trend, 
                    seasonal=seasonal, 
                    seasonal_periods=seasonal_periods
                )
                exp_smooth_model_fit = exp_smooth_model.fit()
                
                # Calculate RMSE to evaluate the model
                y_pred_train = exp_smooth_model_fit.fittedvalues
                rmse = np.sqrt(((y_train - y_pred_train) ** 2).mean())  # RMSE calculation
                
                # Select the best model based on RMSE
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = exp_smooth_model_fit
            except Exception as e:
                continue  # In case of any errors (e.g., singular matrix)

        # Forecast from the forecast start date to the forecast end date
        forecast_dates = pd.date_range(forecast_period_start, forecast_period_end, freq='D')
        y_pred_es = best_model.forecast(len(forecast_dates))

        # Create a DataFrame for the forecast predictions
        forecast_df = pd.DataFrame({
            "Sale Date": forecast_dates,
            "Predicted Yield (Exp Smoothing)": y_pred_es
        })

        # --- Create a table for the forecast predictions ---
        # Now merge this forecasted data with sector-wise averages
        forecast_df['Sector'] = df_filtered['Sector'].iloc[0]  # Assign sector for prediction
        final_forecast = forecast_df.groupby('Sector').agg(
            Avg_YLD_USD_Forecast=("Predicted Yield (Exp Smoothing)", "mean")
        ).reset_index()

        # Save to Excel
        output = pd.ExcelWriter("forecasted_yield_by_sector.xlsx", engine="xlsxwriter")
        final_forecast.to_excel(output, index=False, sheet_name="Forecast")
        
        # Save the Excel file for download
        output.save()
        
        # Display download link for the Excel file
        st.download_button(
            label="Download Forecast Excel Sheet",
            data=open("forecasted_yield_by_sector.xlsx", "rb").read(),
            file_name="forecasted_yield_by_sector.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
