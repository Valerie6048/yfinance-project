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

news = stock.news

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

def analyze_sentiment(url):
    # Mengambil teks artikel dari URL
    article = Article(url)
    article.download()
    article.parse()
    text = article.text

    # Menganalisis sentimen menggunakan TextBlob
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity

    # Menentukan label sentimen
    if sentiment_score >= 0.6:
        sentiment_label = "Sangat Positif"
    elif 0.3 <= sentiment_score < 0.6:
        sentiment_label = "Positif"
    elif 0.1 <= sentiment_score < 0.3:
        sentiment_label = "Agak Positif"
    elif -0.1 <= sentiment_score < 0.1:
        sentiment_label = "Netral"
    elif -0.3 <= sentiment_score < -0.1:
        sentiment_label = "Agak Negatif"
    elif -0.6 <= sentiment_score < -0.3:
        sentiment_label = "Negatif"
    else:
        sentiment_label = "Sangat Negatif"

    return sentiment_score, sentiment_label,text

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

    urls = [item['link'] for item in news]
    titles = [item['title'] for item in news]

    articles_info = [{'news title': title, 'news url': url} for title, url in zip(titles, urls)]

    data=[]

    for article in articles_info:
        sentiment_score, sentiment_label, text = analyze_sentiment(article['news url'])
        article['skor_sentiment'] = sentiment_score
        article['label_sentiment'] = sentiment_label
        data.append(article)
    
    df = pd.DataFrame(data)

    st.header("Sentiment Analysis")
    st.write(df)


