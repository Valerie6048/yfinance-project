import streamlit as st

from datetime import datetime, timedelta

import pandas as pd
import pandas as ta
import json
import numpy as np

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import yfinance as yf

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

st.set_option('deprecation.showPyplotGlobalUse', False)

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

icon("📈")
"""
# Stock Market Analysis Project
## Overview
In this Projec I want to make an Analysis, Visualization, Prediction, and Sentiment Analysis of stock market data using several library like plotly, statsmodels, pandas, etc.
"""
stockToken = st.text_input('Insert the Stock Token', 'NVDA')
stock = yf.Ticker(stockToken)

def get_first_trading_date(symbol):
    # Membuat objek ticker
    ticker = yf.Ticker(symbol)

    # Mendapatkan riwayat harga
    history = ticker.history(period="max")

    # Mengambil tanggal pertama saat harga tersedia
    first_trading_date = history.index[0]

    return first_trading_date

def get_last_trading_date(symbol):
    # Creating a Ticker object
    ticker = yf.Ticker(symbol)

    # Getting historical prices
    history = ticker.history(period="max")

    # Getting the last trading date
    last_trading_date = history.index[-1]

    return last_trading_date

def get_company_name(symbol):
    # Creating a Ticker object
    ticker = yf.Ticker(symbol)

    # Getting company info
    company_info = ticker.info

    # Getting company name
    company_name = company_info['longName']

    return company_name

with st.sidebar:
    st.title('Biodata')
    """
    Name: Akhmad Nizar Zakaria
    
    Github: [Valerie6048](https://github.com/Valerie6048)
    
    LinkedIn: [Akhmad Nizar Zakaria](https://www.linkedin.com/in/akhmad-nizar-zakaria-8a692b229/)

    """
    st.caption('@Valerie6048')

tabs1, tabs2, tabs3 = st.tabs(["Company Description and Analyisis", "Data Visualiation and Prediction", "Technical Analysis"])

with tabs1:
    st.header("Company Description and Analysis")
    corpName = stock.info['longName']
    website = stock.info['website']
    industryType = stock.info['industry']
    industrySector = stock.info['sector']
    businessSummary = stock.info['longBusinessSummary']
    currentPrice = stock.info['currentPrice']
    targetHigh = stock.info['targetHighPrice']
    targetLow = stock.info['targetLowPrice']
    targetMean = stock.info['targetMeanPrice']
    targetMedian = stock.info['targetMedianPrice']
    recommendationScore = stock.info['recommendationMean']
    recommendationKey = stock.info['recommendationKey']

    markdown_text_left = f"""
    ### Company Name
    {corpName}

    ### Company Type
    {industryType}

    ### Company Sector
    {industrySector}
    
    ### Website
    {website}
    """

    markdown_text_right = f"""
    ### Recommendation Score
    {recommendationScore}

    ### Recommendation Key
    {recommendationKey}

    ### Current Price
    ${currentPrice}

    ### Target High Price
    ${targetHigh}

    ### Target Low Price
    ${targetLow}
    """
    left_column, right_column = st.columns(2)

    with left_column:
        st.markdown(markdown_text_left)

    with right_column:
        st.markdown(markdown_text_right)
    
    markdown_text_center = f"""
    ### Company Detail
    {businessSummary}
    """

    st.markdown(markdown_text_center)

with tabs2:
    st.header("Stock Market Chart")
    st.subheader("Chart")

    company_name = get_company_name(stockToken)
    first_trading_date = get_first_trading_date(stockToken)

    end_date = pd.Timestamp.today(tz='America/New_York').ceil('D')
    start_date = first_trading_date
    data = stock.history(start=start_date,end=end_date, interval='1d').reset_index()

    years_difference = (end_date - start_date).days / 365

    fig = go.Figure(data=go.Scatter(
        x=data['Date'],
        y=data['Close'],
        mode='lines',
        name='Close Price'
    ))
    fig.update_layout(
        title=company_name,
        title_x=0.5,
        autosize=False,
        width=900,
        height=500,
        xaxis=dict(rangeselector=dict(
            buttons=list([
                dict(count=30,
                        label="30D",
                        step="day",
                        stepmode="backward"),
                dict(count=6,
                        label="6M",
                        step="month",
                        stepmode="backward"),
                dict(count=1,
                        label="YTD",
                        step="year",
                        stepmode="todate"),
                dict(count=1,
                        label="1Y",
                        step="year",
                        stepmode="backward"),
                dict(count=3,
                        label="3Y",
                        step="year",
                        stepmode="backward"),
                dict(count=years_difference,
                        label="MAX",
                        step="year",
                        stepmode="backward")
            ])
        )),
    )
    st.plotly_chart(fig)

    end_date = pd.Timestamp.today(tz='America/New_York').ceil('D')
    start_date = end_date - pd.Timedelta(7,'D') # Get the last 4 days, in case of holidays/weekend
    data = stock.history(start=start_date, end=end_date, interval='15m').reset_index()
    data = data.rename(columns=dict(Datetime='Date'))
    data = data.loc[data.Date.dt.date == data.Date.dt.date.max()] # Get only the last day's data

    fig = go.Figure(data=go.Candlestick(
        x = data.Date,
        open = data.Open,
        high = data.High,
        low = data.Low,
        close = data.Close
    ))

    st.subheader("Daily Candle Chart")

    fig.update_layout(
        title=company_name,
        title_x=0.5,
        autosize=False,
        width=800,
        height=600,
        xaxis= dict(rangeselector=dict(
            buttons=list([
                dict(count=1,
                    label="1H",
                    step="hour",
                    stepmode="backward"),
                dict(count=3,
                    label="3H",
                    step="hour",
                    stepmode="backward"),
                dict(label='1D',step="all")
            ])
        )),
    )
    st.plotly_chart(fig)

with tabs3:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=120)  # 4 months before today
    stock_data = yf.download(stockToken, start=start_date, end=end_date)

    # Calculate technical indicators using pandas-ta
    stock_data.ta.macd(append=True)
    stock_data.ta.rsi(append=True)
    stock_data.ta.bbands(append=True)
    stock_data.ta.obv(append=True)

    # Calculate additional technical indicators
    stock_data.ta.sma(length=20, append=True)
    stock_data.ta.ema(length=50, append=True)
    stock_data.ta.stoch(append=True)
    stock_data.ta.adx(append=True)

    # Calculate other indicators
    stock_data.ta.willr(append=True)
    stock_data.ta.cmf(append=True)
    stock_data.ta.psar(append=True)

    #convert OBV to million
    stock_data['OBV_in_million'] =  stock_data['OBV']/1e7
    stock_data['MACD_histogram_12_26_9'] =  stock_data['MACDh_12_26_9'] # not to confuse chatGTP

    # Summarize technical indicators for the last day
    last_day_summary = stock_data.iloc[-1][['Adj Close',
        'MACD_12_26_9','MACD_histogram_12_26_9', 'RSI_14', 'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0','SMA_20', 'EMA_50','OBV_in_million', 'STOCHk_14_3_3',
        'STOCHd_14_3_3', 'ADX_14',  'WILLR_14', 'CMF_20',
        'PSARl_0.02_0.2', 'PSARs_0.02_0.2'
    ]]

    # Plot the technical indicators
    fig, axs = plt.subplots(3, 3, figsize=(14, 8))

    # Price Trend Chart
    axs[0, 0].plot(stock_data.index, stock_data['Adj Close'], label='Adj Close', color='blue')
    axs[0, 0].plot(stock_data.index, stock_data['EMA_50'], label='EMA 50', color='green')
    axs[0, 0].plot(stock_data.index, stock_data['SMA_20'], label='SMA_20', color='orange')
    axs[0, 0].set_title("Price Trend")
    axs[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))  # Format date as "Jun14"
    axs[0, 0].tick_params(axis='x', rotation=45, labelsize=8)  # Adjust font size
    axs[0, 0].legend()

    # On-Balance Volume Chart
    axs[0, 1].plot(stock_data['OBV'], label='On-Balance Volume')
    axs[0, 1].set_title('On-Balance Volume (OBV) Indicator')
    axs[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))  # Format date as "Jun14"
    axs[0, 1].tick_params(axis='x', rotation=45, labelsize=8)  # Adjust font size
    axs[0, 1].legend()

    # MACD Plot
    axs[0, 2].plot(stock_data['MACD_12_26_9'], label='MACD')
    axs[0, 2].plot(stock_data['MACDh_12_26_9'], label='MACD Histogram')
    axs[0, 2].set_title('MACD Indicator')
    axs[0, 2].xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))  # Format date as "Jun14"
    axs[0, 2].tick_params(axis='x', rotation=45, labelsize=8)  # Adjust font size
    axs[0, 2].legend()

    # RSI Plot
    axs[1, 0].plot(stock_data['RSI_14'], label='RSI')
    axs[1, 0].axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
    axs[1, 0].axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
    axs[1, 0].legend()
    axs[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))  # Format date as "Jun14"
    axs[1, 0].tick_params(axis='x', rotation=45, labelsize=8)  # Adjust font size
    axs[1, 0].set_title('RSI Indicator')

    # Bollinger Bands Plot
    axs[1, 1].plot(stock_data.index, stock_data['BBU_5_2.0'], label='Upper BB')
    axs[1, 1].plot(stock_data.index, stock_data['BBM_5_2.0'], label='Middle BB')
    axs[1, 1].plot(stock_data.index, stock_data['BBL_5_2.0'], label='Lower BB')
    axs[1, 1].plot(stock_data.index, stock_data['Adj Close'], label='Adj Close', color='brown')
    axs[1, 1].set_title("Bollinger Bands")
    axs[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))  # Format date as "Jun14"
    axs[1, 1].tick_params(axis='x', rotation=45, labelsize=8)  # Adjust font size
    axs[1, 1].legend()

    # Stochastic Oscillator Plot
    axs[1, 2].plot(stock_data.index, stock_data['STOCHk_14_3_3'], label='Stoch %K')
    axs[1, 2].plot(stock_data.index, stock_data['STOCHd_14_3_3'], label='Stoch %D')
    axs[1, 2].set_title("Stochastic Oscillator")
    axs[1, 2].xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))  # Format date as "Jun14"
    axs[1, 2].tick_params(axis='x', rotation=45, labelsize=8)  # Adjust font size
    axs[1, 2].legend()

    # Williams %R Plot
    axs[2, 0].plot(stock_data.index, stock_data['WILLR_14'])
    axs[2, 0].set_title("Williams %R")
    axs[2, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))  # Format date as "Jun14"
    axs[2, 0].tick_params(axis='x', rotation=45, labelsize=8)  # Adjust font size

    # ADX Plot
    axs[2, 1].plot(stock_data.index, stock_data['ADX_14'])
    axs[2, 1].set_title("Average Directional Index (ADX)")
    axs[2, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))  # Format date as "Jun14"
    axs[2, 1].tick_params(axis='x', rotation=45, labelsize=8)  # Adjust font size

    # CMF Plot
    axs[2, 2].plot(stock_data.index, stock_data['CMF_20'])
    axs[2, 2].set_title("Chaikin Money Flow (CMF)")
    axs[2, 2].xaxis.set_major_formatter(mdates.DateFormatter('%b%d'))  # Format date as "Jun14"
    axs[2, 2].tick_params(axis='x', rotation=45, labelsize=8)  # Adjust font size

    # Show the plots
    plt.tight_layout()
    st.pyplot(fig)



st.caption("@Valerie6048")