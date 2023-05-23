
# define dependencies and libraries to be used
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from yahoo_fin.stock_info  import get_live_price, get_stats, get_analysts_info

from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go



### Define functions

#Create a function to minimize, to find the portfolio with the highest sharpe Ratio, the highest risk-adjusted return
def sharpe_ptf(W, returns):
    
    ptf_risk = W.dot(returns.cov()).dot(W) ** 0.5 
    #calculating the portfolio risk, the portfolio standard deviation. 

    #calculating the sharpe ratio for the portfolio
    SR = W.dot(returns.mean()) / ptf_risk

    return -SR  #return negative value of the sharpe ratio in order to minimize it. 

def ptf_optimization(stocks, commodities, start, short):

    assets = stocks + commodities
    tickers = assets
    df = yf.download(tickers, start = start)['Adj Close']

    ret_df = np.log(df/df.shift(1)) #calculate log returns for the selected assets
    
    # initial guess: all portfolios with equal weights
    weights = np.ones(len(ret_df.columns))/np.ones(len(ret_df.columns)).sum()

    if short:
        const = ({'type' : 'eq', 'fun' : lambda x: np.sum(x) - 1})
        #now minimize
        results = minimize(sharpe_ptf, weights, ret_df, constraints = const)  
    else:
        # Optimization with positive weights (just long, no short positions)
        const_pos = [{'type' : 'eq', 'fun' : lambda x: np.sum(x) - 1}, 
                    {'type' : 'ineq', 'fun' : lambda x: np.min(x)}]
        results = minimize(sharpe_ptf, weights, ret_df, constraints = const_pos)

    return results['x'] #return an array with weights of the ptf

def get_STOCK_DATA(stock, start_date):

    data = yf.download(stock, start = start_date) #getting data from Yahoo finance

    live_price = get_live_price(stock) #getting the live price

    stats = get_stats(stock)  #more statistics to be included in the dashboard
    an_info = get_analysts_info(stock) #returns a dictionary with analyst estimates

    return data, live_price, stats, an_info

def apply_indicator(indicator, data, window):
    if indicator == 'Simple moving average':
        sma = SMAIndicator(data['Close'], window).sma_indicator()
        return pd.DataFrame({"Close" : data['Close'], "SMA" : sma}), False
    elif indicator == 'Exponential moving average':
        ema = EMAIndicator(data['Close'], window).ema_indicator()
        return pd.DataFrame({"Close" : data['Close'], "EMA" : ema}), False
    elif indicator == 'Relative strength index':
        rsi = RSIIndicator(data['Close']).rsi()
        return pd.DataFrame({"Close" : data['Close'], "RSI" : rsi}), True


### Initialise S&P500, Nasdaq, Dow-Jones, FTSEMIB list of companies and  and Commodities 
# Set the tickers for important commodities futures that can be traded
commodities = {'Gold' : 'GC=F', 'Oil' : 'CL=F', 'Natural gas' : 'NG=F', 'Silver' : 'SI=F', 'Wheat' : 'KE=F'}
index_composites = {'SP500' : '^GSPC', 'FTSEMIB' : 'FTSEMIB.MI', 'NASDAQ' : '^IXIC'}

# Get tickers for SP500, Nasdaq, FTSEMIB
SP500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
SP500 = SP500[0] #take just the first table from the webpage
FTSEMIB = pd.read_html('https://en.wikipedia.org/wiki/FTSE_MIB')
FTSEMIB = FTSEMIB[1]
NASDAQ = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100#Components')
NASDAQ = NASDAQ[4]

#Create a dictionary containing the company name and its ticker
ticks_SP500 = dict(zip(SP500['Security'], SP500['Symbol']))
ticks_FTSE = dict(zip(FTSEMIB['Company'], FTSEMIB['Ticker']))
ticks_NASDAQ = dict(zip(NASDAQ['Company'], NASDAQ['Ticker']))
###



###### Streamlit page configuration

### Config page layout
st.set_page_config(page_icon = "", layout="wide", initial_sidebar_state="auto", menu_items=None)
st.title("Stock analysis, price prediction and portfolio optimization Dashboard")
st.header("")

### First row 
col1, col2, col3 = st.columns(3)    #form avoid to re-run the script automatically everytime the user change an input value. 
        
with col1:
    market = st.selectbox("Which market index are you interested in?", ('S&P500', 'NASDAQ', 'FTSEMIB', 'Commodities', "Indexes' composites") )
    st.write('You selected:', market)
    if market == 'NASDAQ':
        tickers = ticks_NASDAQ #return and assign all the companies listed in the nasdaq
    elif market == 'FTSEMIB':
        tickers = ticks_FTSE #companies listed in the dowjones
    elif market == 'S&P500':
        tickers = ticks_SP500 #companies listed in the sp500
    elif market == 'Commodities':
        tickers = commodities
    else:
        tickers = index_composites
    
with col2:
    name = st.selectbox('Which stock are you interested in?', list(tickers.keys()))  
    choice = tickers[name]
    st.write('You selected:', choice)

with col3:
    date = st.text_input('Select the starting year:', '2000-01-01')
    st.write('The current selected starting date is', date)


### CREATE THE SECOND ROW, containing live stock price.
col21, col22 = st.columns([3,10]) 
with col21:
    st.caption("") 
    st.header(f"{name} ({choice})")
    st.caption("")
with col22:
    #Creating a live price display in the page. 
    st.caption("")   
    st.metric(label = choice + " live stock price", value = "%.2f$" % get_live_price(choice))

### Second row
stock_data = yf.download(choice, start=date)

indicators = ['Simple moving average', 'Exponential moving average', 'Relative strength index']
    
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Chart", "Technical analysis", "Prediction", "Statistics", "Analysts estimates"])
st.caption("")

with tab1:
    fig1 = px.line(stock_data['Close'])  #define a plotly lineplot for the timeseries of the stockprice
    fig1.update_xaxes(nticks = 20) #set the number of ticks for the x axis
    fig1.update_layout(height = 600) #change the container height
    st.plotly_chart(fig1, use_container_width = True)

    st.caption("")
    st.write(stock_data) 

with tab2:
    
    st.caption("")
    indicator = st.selectbox('Select a technical indicator (only for Moving Averages)', indicators)
    window = st.slider('Select a time window in days', value = 30)

    ind_data, if_rsi = apply_indicator(indicator, stock_data, window) #using apply indicator function we'll get the technical indicator data

    if if_rsi: # when the indicator is RSI we'll plot Close price and RSI in separate plots
        fig2 = make_subplots(rows=2, cols=1, vertical_spacing=0.065, shared_xaxes=True) #we want two subplots in the same figure
        #by setting shared axis True plotly will allow us to interact with one plot but see che changes also in the other

        fig2.add_trace(
            go.Scatter(x=ind_data.index, y=ind_data['Close'], name = 'Close price'),
            row=1, col=1
        )
        fig2.add_trace(
            go.Scatter(x=ind_data.index, y=ind_data['RSI'], name = 'RSI'),
            row=2, col=1
        )
        fig2['layout']['yaxis']['title']='Close price'
        fig2['layout']['yaxis2']['title']='RSI percentage'
        fig2.update_layout(height = 600) #update height to improve the readability
        st.plotly_chart(fig2, use_container_width=True) # this is the streamlit command to plot a plotly object
    else: 
        fig2 = px.line(ind_data)
        fig2.update_xaxes(nticks = 20) #set the number of ticks for the x axis
        fig2.update_layout(height = 600) 
        fig2.update_layout(yaxis_title='Close price')
        st.plotly_chart(fig2, use_container_width=True)
        
        




