
# define dependencies and libraries to be used
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from yahoo_fin.stock_info  import get_data, get_live_price, get_stats, tickers_nasdaq, tickers_sp500, tickers_dow, get_analysts_info, get_quote_data

import streamlit as st
import yfinance as yf


# Create functions: 

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


#def get_sp500_components():
#    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
#    df = df[0]
#    tickers = df["Symbol"].to_list()
#    tickers_companies_dict = dict(zip(df["Symbol"], df["Security"]))
#   return tickers, tickers_companies_dict

commodities = {'Gold' : 'GC=F', 'Oil' : 'CL=F', 'Natural gas' : 'NG=F'}


with st.form(key='my_form'):
    col1, col2, col3 = st.columns(3)    #form avoid to re-run the script automatically everytime the user change an input value. 
         
    with col1:
        market = st.selectbox("Which market index are you interested in?", ('NASDAQ', 'DOWJONES', 'S&P500', 'Commodities') )
        st.write('You selected:', market)
        if market == 'NASDAQ':
            tickers = tickers_nasdaq() #return and assign all the companies listed in the nasdaq
        elif market == 'DOWJONES':
            tickers = tickers_dow() #companies listed in the dowjones
        elif market == 'S&P500':
            tickers = tickers_sp500() #companies listed in the sp500
        else:
            tickers = commodities

    with col2:
        choice = st.selectbox('Which stock are you interested in?', list(tickers.keys()))  
        st.write('You selected:', choice)

    with col3:
        #select the starting year
        year = st.text_input('Select the starting year:', '2000')
        st.write('The current selected starting year is', year)

    submit_button = st.form_submit_button(label='Submit')  #defining the SUBMIT BUTTON AFTER THE THREE SELECTBOXES
        
