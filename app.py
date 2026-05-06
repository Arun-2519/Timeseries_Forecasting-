import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("sales_model.pkl")

# Features used during training
features = ['Size','Sleeve','Length','Range','Mrp','Month']

st.title("📊 Sales Forecasting App")

days = st.number_input("Enter number of future days (e.g., 7 or 30):", min_value=1, max_value=365, value=7)

size = st.number_input("Size:", min_value=1, value=2)
sleeve = st.number_input("Sleeve:", min_value=0, value=1)
length = st.number_input("Length:", min_value=1, value=1)
range_ = st.number_input("Range:", min_value=1, value=2)
mrp = st.number_input("MRP:", min_value=100, value=1000)

last_date = pd.to_datetime("2026-05-06")

if st.button("Predict"):
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days)

    future_rows = []
    for d in future_dates:
        row = {
            'Size': size,
            'Sleeve': sleeve,
            'Length': length,
            'Range': range_,
            'Mrp': mrp,
            'Month': d.month
        }
        future_rows.append(row)

    future_df = pd.DataFrame(future_rows)
    y_future = model.predict(future_df[features])
    future_df['Predicted_Qty'] = y_future
    future_df['Date'] = future_dates

    st.write("### Forecast Results")
    st.dataframe(future_df[['Date','Month','Predicted_Qty']])
    st.line_chart(future_df.set_index('Date')['Predicted_Qty'])
