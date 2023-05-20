
# define dependencies and libraries to be used
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from yahoo_fin.stock_info  import get_data, get_live_price, get_stats, tickers_nasdaq, tickers_sp500, tickers_dow, get_analysts_info

import streamlit as st
import yfinance as yf


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

def get_STOCK_DATA(stock, year):

    data = get_data(stock, start_date = year) #using the get_data function I am getting the data from Yahoo finance
    data.index.name = 'date'  #setting index name as date
    data.reset_index(inplace=True)  #when resetting the index, the old index with dates is added as a column in the dataframe
    data['date'] = data['date'].dt.date  #fixing the new date column format 

    live_price = get_live_price(stock) #getting the live price

    stats = get_stats(stock)  #more statistics to be included in the dashboard
    an_info = get_analysts_info(stock) #returns a dictionary with analyst estimates

    return data, live_price, stats, an_info


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

ticks_SP500 = dict(zip(SP500['Security'], SP500['Symbol']))
ticks_FTSE = dict(zip(FTSEMIB['Company'], FTSEMIB['Ticker']))
ticks_NASDAQ = dict(zip(NASDAQ['Company'], NASDAQ['Ticker']))
###



###### Streamlit page configuration

### Config page layout
st.set_page_config(page_icon = "", layout="wide", initial_sidebar_state="auto", menu_items=None)
st.title("Stock analysis, price prediction and portfolio optimization Dashboard")
st.subheader("")

### First row 
with st.form(key='my_form'):
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
        #select the starting year
        year = st.text_input('Select the starting year:', '2000')
        st.write('The current selected starting year is', year)

    submit_button = st.form_submit_button(label='Submit')  #defining the SUBMIT BUTTON AFTER THE THREE SELECTBOXES
        
if submit_button: #if submit button is pressed, the rest of the script can be executed (only the first time)
    
    stock_data, stock_live_price, stats, analyst_info = get_STOCK_DATA(choice, year)

    st.caption("")
    st.write(stock_data) 