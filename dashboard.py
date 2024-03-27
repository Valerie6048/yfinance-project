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
