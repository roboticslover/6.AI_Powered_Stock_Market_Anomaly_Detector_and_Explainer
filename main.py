import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Stock Market Anomaly Detector",
    layout="wide"
)

# Initialize cache file
CACHE_FILE = "explanations_cache.json"

# Load cached explanations
@st.cache_resource
def get_cached_explanations():
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_to_cache(cache_data):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f)

# Function to get AI explanation
def get_ai_explanation(symbol, date, price, volume, price_change, volume_change, price_z_score, volume_z_score):
    """
    Generate an AI explanation for a stock anomaly using OpenAI's API.
    """
    try:
        # Get the OpenAI API key from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("OpenAI API key not found. Please set it in the .env file.")
            return "OpenAI API key not found."

        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)

        # Construct the message for ChatCompletion
        messages = [
            {"role": "system", "content": "You are a financial analyst providing detailed explanations for stock market anomalies."},
            {"role": "user", "content": (
                f"On {date}, the stock {symbol} had a closing price of ${price:.2f} "
                f"and a trading volume of {volume:,}. The price changed by {price_change:+.2f}% "
                f"and the volume changed by {volume_change:+.2f}% compared to the previous day. "
                f"The price Z-score was {price_z_score:.2f} and the volume Z-score was {volume_z_score:.2f}. "
                f"What could be the possible reasons for this anomaly? "
                f"Consider company news, market trends, or global events that could have impacted the stock."
            )}
        ]

        # Call the OpenAI ChatCompletion API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )

        # Extract the explanation
        explanation = response.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        return f"Error generating AI explanation: {e}"

# App title
st.title("Stock Market Anomaly Detector")

# Sidebar inputs
st.sidebar.header("Analysis Parameters")

def user_input_features():
    stock_symbol = st.sidebar.text_input("Stock Symbol", 'AAPL')
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2023-01-01'))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime('today'))
    z_score_threshold = st.sidebar.slider(
        "Anomaly Detection Threshold (Z-Score)",
        min_value=2.0,
        max_value=4.0,
        value=2.5,
        step=0.1,
        help="Higher values mean fewer but more significant anomalies"
    )
    return stock_symbol, start_date, end_date, z_score_threshold

stock_symbol, start_date, end_date, z_score_threshold = user_input_features()

# Function to fetch data
@st.cache_data(ttl=3600)
def fetch_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)
        if data.empty:
            st.error("No data fetched. Check the stock symbol and date range.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to plot data with anomalies highlighted
def plot_data_with_anomalies(data, anomalies):
    # Closing Price with anomalies
    fig_close, ax_close = plt.subplots(figsize=(10, 4))
    ax_close.plot(data['Close'], label='Closing Price', color='blue')
    if not anomalies.empty:
        ax_close.scatter(anomalies.index, anomalies['Close'], color='red', label='Anomalies')
    ax_close.set_title('Closing Price with Anomalies')
    ax_close.set_xlabel('Date')
    ax_close.set_ylabel('Price ($)')
    ax_close.legend()
    st.pyplot(fig_close)

    # Volume with anomalies
    fig_volume, ax_volume = plt.subplots(figsize=(10, 4))
    ax_volume.plot(data['Volume'], label='Volume', color='green')
    if not anomalies.empty:
        ax_volume.scatter(anomalies.index, anomalies['Volume'], color='red', label='Anomalies')
    ax_volume.set_title('Volume with Anomalies')
    ax_volume.set_xlabel('Date')
    ax_volume.set_ylabel('Volume')
    ax_volume.legend()
    st.pyplot(fig_volume)

# Function to detect anomalies
def detect_anomalies(data, z_score_threshold):
    # Calculate rolling statistics
    window = 20
    data['Price_Mean'] = data['Close'].rolling(window=window).mean()
    data['Price_Std'] = data['Close'].rolling(window=window).std()
    data['Volume_Mean'] = data['Volume'].rolling(window=window).mean()
    data['Volume_Std'] = data['Volume'].rolling(window=window).std()

    # Calculate Z-scores
    data['Price_Z_Score'] = (data['Close'] - data['Price_Mean']) / data['Price_Std']
    data['Volume_Z_Score'] = (data['Volume'] - data['Volume_Mean']) / data['Volume_Std']

    # Calculate daily percentage changes
    data['Price_Change'] = data['Close'].pct_change() * 100
    data['Volume_Change'] = data['Volume'].pct_change() * 100

    # Detect anomalies
    anomalies = data[
        (data['Price_Z_Score'].abs() > z_score_threshold) |
        (data['Volume_Z_Score'].abs() > z_score_threshold)
    ]
    return anomalies

# Function to generate statistical explanation
def generate_statistical_explanation(row):
    explanation = f"""
    **Statistical Analysis:**
    - Closing Price: ${row['Close']:.2f}
    - Volume: {int(row['Volume']):,}
    - Price Change: {row['Price_Change']:+.2f}%
    - Volume Change: {row['Volume_Change']:+.2f}%
    - Price Z-score: {row['Price_Z_Score']:.2f}
    - Volume Z-score: {row['Volume_Z_Score']:.2f}
    """
    return explanation

# Main execution
def main():
    st.sidebar.info("This app detects stock market anomalies and provides explanations.")

    data = fetch_data(stock_symbol, start_date, end_date)
    if data is None:
        return

    anomalies = detect_anomalies(data, z_score_threshold)

    # Display market statistics
    st.subheader("Market Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Price", f"${data['Close'].mean():.2f}")
    col2.metric("Price Volatility", f"${data['Close'].std():.2f}")
    col3.metric("Anomalies Detected", len(anomalies))

    # Plot data
    st.subheader("Price and Volume Charts")
    plot_data_with_anomalies(data, anomalies)

    # Display anomalies
    if not anomalies.empty:
        st.subheader("Detected Anomalies")
        cached_explanations = get_cached_explanations()
        explanations_updated = False

        for index, row in anomalies.iterrows():
            date_str = index.strftime('%Y-%m-%d')
            st.write(f"### {date_str}")
            st.write(generate_statistical_explanation(row))

            # Generate AI explanation
            cache_key = f"{stock_symbol}_{date_str}"
            explanation = cached_explanations.get(cache_key)
            if explanation:
                # Display cached explanation
                st.write("**AI Explanation (Cached):**")
                st.write(explanation)
            else:
                # Add "Explain" button
                if st.button(f"Explain Anomaly on {date_str}", key=date_str):
                    with st.spinner("Generating AI explanation..."):
                        explanation = get_ai_explanation(
                            stock_symbol,
                            date_str,
                            row['Close'],
                            row['Volume'],
                            row['Price_Change'],
                            row['Volume_Change'],
                            row['Price_Z_Score'],
                            row['Volume_Z_Score']
                        )
                        st.write("**AI Explanation:**")
                        st.write(explanation)
                        # Cache the explanation
                        cached_explanations[cache_key] = explanation
                        explanations_updated = True

        # Save cached explanations if updated
        if explanations_updated:
            save_to_cache(cached_explanations)
    else:
        st.write("No anomalies detected with the current threshold.")

    # Export anomalies to CSV
    if not anomalies.empty:
        st.subheader("Download Anomaly Data")
        csv = anomalies.to_csv()
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{stock_symbol}_anomalies.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()