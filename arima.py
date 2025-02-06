import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
from itertools import product

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

    # Get unique sectors and flight dates for user selection
    sectors = df['Sector'].unique()
    selected_sector = st.selectbox('Select Sector', sectors)

    flight_dates = df['Flight Date'].unique()
    departure_date = st.selectbox('Select Departure Date', flight_dates)
    departure_date = pd.to_datetime(departure_date)

    # Calculate the forecast period window (10 days before departure date)
    forecast_window_start = departure_date - pd.Timedelta(days=90)
    forecast_window_end = departure_date

# Display the valid range for the forecast start and end dates
    #st.write(f"Forecast period must be between {forecast_window_start.date()} and {forecast_window_end.date()}.")

# Section 2: Select the forecast start date
    forecast_period_start = st.date_input(
        "Select Forecast Start Date",
        min_value=forecast_window_start,
        max_value=forecast_window_end
    )

# Section 3: Select the forecast end date
    forecast_period_end = st.date_input(
        "Select Forecast End Date",
        min_value=forecast_period_start,  # Ensure the end date is after or equal to the start date
        max_value=forecast_window_end     # Ensure the end date is before or equal to the departure date
    )

# Validate the selected range and display it
    if forecast_period_start and forecast_period_end:
        st.write(f"Selected Forecast Period: {forecast_period_start} to {forecast_period_end}")

    # Add a button to generate the plots
    if st.button("Generate Forecast Plots"):
        # Filter data based on the selected sector
        df_filtered = df[df['Sector'] == selected_sector]

        # Section 2: Group and create features
        df_grouped = df_filtered.groupby("Sale Date", as_index=False).agg(
            Avg_YLD_USD=("YLD USD", "mean"),
            Sum_PAX=("PAX COUNT", "sum")
        )

        df_grouped['Days_Before_Dep'] = (departure_date - df_grouped["Sale Date"]).dt.days
        df_grouped = df_grouped[df_grouped['Days_Before_Dep'] > 0]

        df_grouped['Days_Before_Dep_Squared'] = df_grouped['Days_Before_Dep'] ** 2
        df_grouped['PAX_YLD_Product'] = df_grouped['Sum_PAX'] * df_grouped['Avg_YLD_USD']

        for lag in [1, 2, 3, 5, 7]:  # Add lag features for selected days
            df_grouped[f"Lag_{lag}"] = df_grouped["Avg_YLD_USD"].shift(lag)

        # Add Moving Averages
        df_grouped["MA_3"] = df_grouped["Avg_YLD_USD"].rolling(window=3).mean()
        df_grouped["MA_7"] = df_grouped["Avg_YLD_USD"].rolling(window=7).mean()

        # Add Exponential Weighted Moving Averages
        df_grouped["EWMA_3"] = df_grouped["Avg_YLD_USD"].ewm(span=3, adjust=False).mean()
        df_grouped["EWMA_7"] = df_grouped["Avg_YLD_USD"].ewm(span=7, adjust=False).mean()

        # Drop NaNs caused by shifting and rolling
        df_grouped = df_grouped.dropna()

        # Section 3: Time series plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_grouped['Sale Date'],
            y=df_grouped['Avg_YLD_USD'],
            mode='lines',
            name='Average Yield (USD)'
        ))

        fig.update_layout(
            title=f"Time Series of Average Yield for {selected_sector}",
            xaxis_title="Sale Date",
            yaxis_title="Average Yield (USD)",
            template="plotly_dark"
        )

        st.plotly_chart(fig)

        # Section 4: Prepare data for Exponential Smoothing

        # Filter data up to the forecast period start date
        df_forecast_data = df_grouped[df_grouped['Sale Date'] <= pd.Timestamp(forecast_period_start)]

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
                rmse = mean_absolute_error(y_train, y_pred_train)
                
                # Select the best model based on RMSE
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = exp_smooth_model_fit
                    best_params = (trend, seasonal, seasonal_periods)
            except Exception as e:
                continue  # In case of any errors (e.g., singular matrix)

        # Output the best parameters
        #st.write(f"Best model parameters: Trend = {best_params[0]}, Seasonal = {best_params[1]}, Seasonal Periods = {best_params[2]}")

        # Forecast from the forecast start date to the departure date
        forecast_dates = pd.date_range(forecast_period_start, forecast_period_end, freq='D')
        y_pred_es = best_model.forecast(len(forecast_dates))

        # Create a DataFrame for the forecast predictions
        forecast_df = pd.DataFrame({
            "Sale Date": forecast_dates,
            "Predicted Yield (Exp Smoothing)": y_pred_es
        })

        # Merge forecast predictions with actual data (if available)
        actual_data = df_grouped[df_grouped['Sale Date'].isin(forecast_dates)]

        # Create a table with actual and predicted values
        forecast_table = pd.merge(actual_data[['Sale Date', 'Avg_YLD_USD']], forecast_df, on="Sale Date", how="left")
        forecast_table.rename(columns={"Avg_YLD_USD": "Actual Yield (USD)"}, inplace=True)

        # Display the actual vs predicted table
        st.write("Actual vs Predicted Yield Table", forecast_table)

        # Plot the actual vs predicted yields for the forecast period
        fig_pred = go.Figure()

        fig_pred.add_trace(go.Scatter(
            x=df_grouped['Sale Date'],
            y=df_grouped['Avg_YLD_USD'],
            mode='lines',
            name='Actual Yield'
        ))

        fig_pred.add_trace(go.Scatter(
            x=forecast_df["Sale Date"],
            y=forecast_df["Predicted Yield (Exp Smoothing)"],
            mode='lines',
            name="Predicted Yield (Exp Smoothing)",
            line=dict(dash="dash")
        ))

        fig_pred.update_layout(
            title="Actual vs Predicted Average Yield (Exp Smoothing)",
            xaxis_title="Sale Date",
            yaxis_title="Average Yield (USD)",
            template="plotly_dark"
        )

        st.plotly_chart(fig_pred)
        # Create a final table with only departure date and predicted yield
        final_forecast_table = pd.DataFrame({
            "Forcast Date": [forecast_df["Sale Date"].date()],  # Ensure it's in date format
            "Predicted Yield (USD)": forecast_df["Predicted Yield (Exp Smoothing)"]  # Last forecasted value
        })

        # Display the final table
        st.write("Predicted Yield for Forecast Date", final_forecast_table)
