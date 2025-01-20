import streamlit as st
import pandas as pd
#import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import plotly.express as px


st.set_page_config(page_title="Forecast The Fare", layout="wide")

st.title("Fare Forecasting")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Load the Excel file and get available sheet names
    excel_file = pd.ExcelFile(uploaded_file)
    sheet_names = excel_file.sheet_names

    # Add a dropdown to select a sheet
    selected_sheet = st.selectbox("Select a sheet", sheet_names)

    # Load the selected sheet
    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

    # Check if required columns exist
    if 'Sector' not in df.columns or 'Sale Date' not in df.columns or 'YLD USD' not in df.columns or 'PAX COUNT' not in df.columns:
        st.error("The dataset must contain 'Sector', 'Sale Date', 'YLD USD', and 'PAX COUNT' columns.")
    else:
        # Extract unique sectors
        sectors = df['Sector'].unique()

        # Add a dropdown button for selecting a sector
        selected_sector = st.selectbox("Select a sector", sectors)

        # Filter the dataset based on the selected sector
        filtered_df = df[df['Sector'] == selected_sector]

        # Ensure Sale Date is in datetime format
        filtered_df['Sale Date'] = pd.to_datetime(filtered_df['Sale Date'], errors='coerce')
        filtered_df['Flight Date'] = pd.to_datetime(filtered_df['Flight Date'], errors='coerce')

        date = df['Flight Date'].unique()

        # Add a dropdown button for selecting a sector
        departure_date = st.selectbox("Select a daparture date", date )

        #departure_date = pd.to_datetime('2024-11-01')
        filtered_df["Days Before Departure"] = (departure_date - filtered_df["Sale Date"]).dt.days

        # Handle missing Sale Date values
        filtered_df = filtered_df.dropna(subset=['Sale Date'])

        # Aggregate data
        aggregated_data = (
            filtered_df.groupby('Sale Date')
            .agg(
                Average_Yield=('YLD USD', 'mean'),
                Sum_Pax_Count=('PAX COUNT', 'sum'),
                Average_Days_Before_Departure=('Days Before Departure', 'mean')
            )
            .reset_index()
        )

        # Create columns for the tabs
        tab1, tab2 = st.tabs(["📈 Revenue Chart", "🗃 Data Table"])

        with tab2:
            # Display the Data Table
            #st.write(f"Data Table for Sector: {selected_sector}")
            #st.dataframe(filtered_df)

            # Optionally, show the aggregated data if needed
            st.write("Aggregated Data (Average Yield, Pax Count):")
            st.dataframe(aggregated_data)


        with tab1:
            # Plot Revenue Chart: Average Yield and Pax Count
            col1, col2 = st.columns(2)

            with col1:
                # Plot 1: Average Yield
                fig_yield = go.Figure()
                fig_yield.add_trace(
                    go.Scatter(
                        x=aggregated_data['Sale Date'],
                        y=aggregated_data['Average_Yield'],
                        mode='lines+markers',
                        name='Average Yield (USD)',
                        line=dict(color='blue')
                    )
                )
                fig_yield.update_layout(
                    title=f"Average Yield Over Time (Sector: {selected_sector})",
                    xaxis_title="Sale Date",
                    yaxis_title="Average Yield (USD)",
                    template="plotly_white"
                )
                st.plotly_chart(fig_yield)

            with col2:
                # Plot 2: Pax Count
                fig_pax = go.Figure()
                fig_pax.add_trace(
                    go.Scatter(
                        x=aggregated_data['Sale Date'],
                        y=aggregated_data['Sum_Pax_Count'],
                        mode='lines+markers',
                        name='Pax Count',
                        line=dict(color='orange')
                    )
                )
                fig_pax.update_layout(
                    title=f"Pax Count Over Time (Sector: {selected_sector})",
                    xaxis_title="Sale Date",
                    yaxis_title="Pax Count",
                    template="plotly_white"
                )
                st.plotly_chart(fig_pax)

       

            # Train-Test Split (up to November for train, November for test)
            cutoff_date = departure_date 
            train_data = aggregated_data[aggregated_data['Sale Date'] < cutoff_date]
            test_data = aggregated_data[aggregated_data['Sale Date'] >= cutoff_date]

            # Apply Exponential Smoothing on the train data
            model = ExponentialSmoothing(
                train_data['Average_Yield'], 
                trend='add', 
                seasonal='add', 
                seasonal_periods=12  # Assumes monthly data, so seasonality period is 12 months
            )
            model_fit = model.fit()

            # Forecast the next period(s) (November in this case)
            forecast = model_fit.forecast(len(test_data))

            # Plot the training data, test data, and forecasted values
            fig = go.Figure()

            # Add Training Data trace
            fig.add_trace(
                go.Scatter(
                    x=train_data['Sale Date'],
                    y=train_data['Average_Yield'],
                    mode='lines+markers',
                    name='Training Data (Actual)',
                    line=dict(color='blue')
                )
            )

            # Add Test Data trace
            fig.add_trace(
                go.Scatter(
                    x=test_data['Sale Date'],
                    y=test_data['Average_Yield'],
                    mode='lines+markers',
                    name='Test Data (Actual)',
                    line=dict(color='orange')
                )
            )

            # Add Forecasted Data trace
            fig.add_trace(
                go.Scatter(
                    x=test_data['Sale Date'],
                    y=forecast,
                    mode='lines+markers',
                    name='Forecasted Data',
                    line=dict(color='green', dash='dot')
                )
            )

            # Update layout
            fig.update_layout(
                title=f"Training, Test, and Forecasted Average Yield for Sector: {selected_sector}",
                xaxis_title="Sale Date",
                yaxis_title="Average Yield (USD)",
                template="plotly_white"
            )

            st.plotly_chart(fig)

            # Calculate forecast accuracy (optional)
            if not test_data.empty:
                mae = mean_absolute_error(test_data['Average_Yield'], forecast)
                mape = (mean_absolute_percentage_error(test_data['Average_Yield'], forecast)) * 100
                st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
