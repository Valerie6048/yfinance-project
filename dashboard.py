import streamlit as st

from datetime import datetime

import pandas as pd
import json
import numpy as np

import plotly.graph_objects as go

import yfinance as yf

from newspaper import Article
from textblob import TextBlob

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

company_name = get_company_name(stock_token)

first_trading_date = get_first_trading_date(stock_token)

def plot_stock_data(stock_symbol):
    # Fetch stock data
    stock = yf.Ticker(stock_symbol)
    company_name = stock.info['longName']
    first_trading_date = stock.info['firstTradeDate']
    end_date = pd.Timestamp.today(tz='America/New_York').ceil('D')
    start_date = first_trading_date
    data = stock.history(start=start_date, end=end_date, interval='1d').reset_index()

    years_difference = (end_date - start_date).days / 365

    # Create Plotly figure
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
    return fig


with st.sidebar:
    st.title('Biodata')
    """
    Name: Akhmad Nizar Zakaria
    
    Github: [Valerie6048](https://github.com/Valerie6048)
    
    LinkedIn: [Akhmad Nizar Zakaria](https://www.linkedin.com/in/akhmad-nizar-zakaria-8a692b229/)

    """
    st.caption('@Valerie6048')

tabs1, tabs2, tabs3 = st.tabs(["Company Description and Analyisis", "Data Visualiation and Prediction", "Sentiment Analysis"])

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

    ### Website
    {website}

    ### Current Price
    ${currentPrice}

    ### Target High Price
    ${targetHigh}

    ### Target Low Price
    ${targetLow}
    """

    markdown_text_right = f"""
    ### Recommendation Score
    {recommendationScore}

    ### Recommendation Key
    {recommendationKey}

    ### Company Type
    {industryType}
    """
    left_column, right_column = st.columns(2)

    with left_column:
        st.markdown(markdown_text_left)

    with right_column:
        st.markdown(markdown_text_right)

with tabs2:
    st.header("NJAY")

    fig = plot_stock_data(stock)
    st.plotly_chart(fig)
