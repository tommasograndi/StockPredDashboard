
# define dependencies and libraries to be used
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from yahoo_fin.stock_info  import get_live_price, get_stats, get_analysts_info
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas_market_calendars as mcal  
from datetime import datetime, timedelta
import pmdarima as pm
import xgboost as xgb



### Define functions

# 1) ARIMA FUNCTION
def ARIMA_forecast(data, forecast_period):    

    model_autoARIMA = pm.auto_arima(data['Close'], start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
    summary = model_autoARIMA.summary()

    parameters = model_autoARIMA.get_params()['order']
    parameters = {'p, number of Autoregressive terms': parameters[0], 
                    'd, difference order': parameters[1], 
                    'q, number of Moving Average terms' : parameters[2]}

    predicted, confint = model_autoARIMA.predict(n_periods=forecast_period, return_conf_int=True)

    # Calculate the dates of the days in for the forecasted period (market year)
    today = datetime.today() #get today's date
    today = data.index[-1] #get last day 
    end_period = (today + timedelta(days=forecast_period)).strftime('%Y-%m-%d') 
    market_year = mcal.get_calendar('Financial_Markets_US') #using get_calendar we obtain the yearly calendar for US markets
    cal = market_year.schedule(start_date=today.strftime('%Y-%m-%d'), end_date=end_period) #now we oobtain the full schedule
    #And finally convert the schedule into a daterange that will be the dateindex of our forecasted series.
    forecast_dates = mcal.date_range(cal, frequency='1D') 

    return forecast_dates, confint, predicted, parameters, summary
# END OF ARIMA

# 2) XGBOOST FUNCTION
def XGBOOST_forecast(data, forecast_period):  

    # Get only the Close price
    df = data.iloc[:, 3:4]

    ### Create set of features

    df['Past_Ret'] = np.log(df['Close']/df['Close'].shift(1))
    df['Future_ret'] = df['Past_Ret'].shift(-1)
    df['Diff_1'] = df['Close'].diff()
    df['Diff_2'] = df['Close'].diff().diff()
    df['MA5']= df['Close'].rolling(window=5).mean()
    df['MA50']= df['Close'].rolling(window=50).mean()

    # ADD LAGS (3)
    for i in range(1, 4):
        df[f'lag{i}'] = df['Close'].shift(i)

    # Drop past returns (not needed for prediction)
    df.drop(axis=1, labels='Past_Ret', inplace=True)

    # Add calendar features
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype(int)

    # Drop observations with NaN
    df.dropna(inplace=True)

    ### Train/test split  
    test_idx  = int(df.shape[0] * (1-0.1)) #index for the test set
    train_df  = df.iloc[:test_idx+1, :].copy()
    test_df   = df.iloc[test_idx:, :].copy()

    #Split target and features 
    xs = list(list([0]) + list(range(2, 16)))

    y_train = train_df['Future_ret'].copy()
    X_train = train_df.iloc[:, xs].copy()

    y_test  = test_df['Future_ret'].copy()
    X_test  = test_df.iloc[:, xs].copy()

    ### Perform XGBOOST
    model = xgb.XGBRegressor(booster='gbtree',    
                        n_estimators=400,
                        early_stopping_rounds=50,
                        objective='reg:squarederror',
                        max_depth=10,
                        learning_rate=0.01,
                        gamma=0.001
                        )
    model.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)
    
    # Calculate the dates of the days for the forecasted period (market calendar)
    today = X_test.index[-1] #get last day of the training set
    end_period = (today + timedelta(days=forecast_period)).strftime('%Y-%m-%d')  #shift to the end period by 300 days
    market_year = mcal.get_calendar('Financial_Markets_US') #using get_calendar we obtain the yearly calendar for US markets
    cal = market_year.schedule(start_date=today.strftime('%Y-%m-%d'), end_date=end_period) #now we oobtain the full schedule
    #And finally convert the schedule into a daterange that will be the dateindex of our forecasted series.
    forecast_dates = mcal.date_range(cal, frequency='1D')

    # Get future calendar features
    dayofweek= forecast_dates.dayofweek
    quarter = forecast_dates.quarter
    month = forecast_dates.month
    year = forecast_dates.year
    dayofyear = forecast_dates.dayofyear
    day = forecast_dates.day
    weekofyear = forecast_dates.isocalendar().week.astype(int)
    
    ### OUT OF SAMPLE forecasting
    X_oos = X_test.iloc[-51:, :] #pick last 50 observation from the test sample (we need them for moving averages)
    # Predict the first return. 
    pred_oos = model.predict(X_oos.iloc[-1:, :]) 

    # Joint last 50 observation dates of test set with future dates (determined by forecast period) 
    combined = X_oos.index.union(forecast_dates[1:])


    for i in range(1, len(forecast_dates)):

        print(i)

        close = (X_oos.iloc[i-1, 0]) * (1+pred_oos[i-1]) # Close price

        ## Loc will insert a new row of features (for a specific day) at the bottom of the dataframe
        #Create new row with only close in order to calculate MA
        X_oos.loc[i] = [close, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        # calculate MAverages (that we'll insert now)
        MA5 = X_oos.iloc[:, 0].rolling(window=5).mean()
        MA50 = X_oos.iloc[:, 0].rolling(window=50).mean()

        # Now subscript the row with all features
        X_oos.loc[i] = [
            close, # Close price
            # pred_oos[i-1], #Returns (predicted)
            close - (X_oos.iloc[i-1, 0]), #first order difference
            (close - (X_oos.iloc[i-1, 0])) - X_oos.iloc[i-1, 1],#second order difference
            MA5[i],
            MA50[i],
            X_oos.iloc[i-1, 0], #lag1
            X_oos.iloc[i-2, 0], #lag2
            X_oos.iloc[i-3, 0], #lag3
            int(dayofweek[i]), #take same daysofweek as X_test
            int(quarter[i]), #take same quarter as X_test
            int(month[i]), #take same month as X_test
            int(year[i]), #take same year as X_test
            int(dayofyear[i]), #take same dayofyear as X_test
            int(day[i]), #take same dayofmonth as X_test
            int(weekofyear[i]) #take same weekofyear as X_test
        ]

        # Now predict the actual return based on the set of features inserted above. And append it to the array containing the first prediction.
        pred_oos = np.append(pred_oos, [model.predict(X_oos.iloc[i:(i+1), :])])
        # This new prediction will be used in the following ITERATION to calculate the Close price at time t

    # After running the whole loop, this will update the dataframe index with the dates calendar obtained before.
    X_oos.set_index(combined, inplace=True)        

    # And now subset only the predictions from today
    X_oos = X_oos[today:]
    
    return X_oos, train_df, test_df
# end XGBOOST

def get_STOCK_DATA(stock, start_date):

    data = yf.download(stock, start = start_date) #getting data from Yahoo finance
    stats = get_stats(stock)  #more statistics to be included in the dashboard
    an_info = get_analysts_info(stock) #returns a dictionary with analyst estimates

    return data, stats, an_info

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


### Initialise S&P500, Nasdaq, Dow-Jones, FTSEMIB list of companies

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
st.title("Stock analysis and price prediction Dashboard")
st.header("")

### First row 
col1, col2, col3 = st.columns(3)   #create three horizontal columns for three different input boxes
        
with col1:
    market = st.selectbox("Which market index are you interested in?", ('S&P500', 'NASDAQ', 'FTSEMIB') )
    st.write('You selected:', market)
    if market == 'NASDAQ':
        tickers = ticks_NASDAQ #return and assign all the companies listed in the nasdaq
    elif market == 'FTSEMIB':
        tickers = ticks_FTSE #companies listed in the dowjones
    elif market == 'S&P500':
        tickers = ticks_SP500 #companies listed in the sp500
    
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

### Third row, containing the tabs displaying all the informations and the PREDICTION
stock_data, stats, analyst_info = get_STOCK_DATA(choice, start_date=date)

# Initialise indicators
indicators = ['Simple moving average', 'Exponential moving average', 'Relative strength index']

# Create the tabs
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
    indicator = st.selectbox('Select a technical indicator', indicators)
    window = st.slider('Select a time window in days (only for Moving Averages)', value = 30)

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
        fig2.add_hline(
            y=70,
            line_dash="dash",
            row=2, col=1
        )
        fig2.add_hline(
            y=30,
            line_dash="dash",
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

# Forecasting using ARIMA and XGBOOST models    
with tab3:
    
    with st.expander('ARIMA Model'):
            
            # Addign a columns only for display purposes
            colAR1, colAR2 = st.columns([3,10]) 
            with colAR1:
                st.caption("") 
                future = st.slider('Select the time-horizon to perform the forecast (days)', 
                               value = 100, 
                               max_value=500,
                               help='Remember that the longer the horizon, the more unreliable will be the forecast ')
            with colAR2:
                st.caption("")   

            # Recall the ARIMA_forecast function to obtain the predictions and the output of the model
            forecast_dates, confint, predicted, parameters, ARIMAsummary = ARIMA_forecast(stock_data, future)

            # Create a chart to display the prediction
            fig3 = go.Figure([go.Scatter( #add the Upper Bound of the confidence interval
                x=forecast_dates, 
                y=confint[:,1], 
                marker=dict(color="lightgrey"),
                name = 'Upper bound CI',
                showlegend=False
                ),
                go.Scatter( #add the lower Bound of the confidence interval
                x=forecast_dates, # x, then x reversed
                y=confint[:,0], # upper, then lower reversed
                marker=dict(color="lightgrey"),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)', #color to fill in the area
                fill='tonexty', #type of filling
                name = 'Upper bound CI',
                showlegend=False
                ), 
                go.Scatter( #add the Prediction
                    x=forecast_dates, y=predicted, line=dict(color='orange'),
                    mode='lines', name='Prediction')
            ])
            fig3.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Original series', line=dict(color='blue'))) #This is the original series
            fig3.update_layout(
                title=f"Price prediction for {choice}", 
                yaxis_title="Price ($)", 
            )
            fig3.update_layout(height = 600)
            fig3.update_xaxes(nticks = 25)
            st.plotly_chart(fig3, use_container_width=True)

            st.subheader(f'ARIMA({list(parameters.values())[0]},{list(parameters.values())[1]},{list(parameters.values())[2]})')
            st.write(parameters)
    
            st.text(ARIMAsummary)


    with st.expander('XGBOOST Model'):
            st.write()

            #Create a slider box to select the time period
            colXG1, colXG2 = st.columns([3,10]) 
            with colXG1:
                st.caption("") 
                futureXG = st.slider('Select the time-horizon to perform the forecast (days)', 
                               value = 100, 
                               max_value=600,
                               help='Remember that the longer the horizon, the more unreliable will be the forecast ')
            with colXG2:
                st.caption("") 
            
            pred, train_df, test_df = XGBOOST_forecast(stock_data, futureXG)

            # Display the prediction
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=train_df.index, y=train_df['Close'], name='Train'))
            fig4.add_trace(go.Scatter(x=test_df.index, y=test_df['Close'], name='Test'))
            fig4.add_trace(go.Scatter(x=pred.index,  y=pred['Close'],  name='Out of sample prediction'))
            fig4.update_layout(height = 600)
            fig4.update_xaxes(nticks = 25)
            st.plotly_chart(fig4, use_container_width=True)


with tab4:
     st.caption("")
     st.write(stats, use_container_width = True)

with tab5:
     st.caption('')
     with st.expander('EPS Trend'):
         st.write(analyst_info['EPS Trend'])

     with st.expander('Earnings estimate'):
         st.write(analyst_info['Earnings Estimate'])

     with st.expander('Growth estimate'):
         st.write(analyst_info['Growth Estimates'].dropna('columns'))

     with st.expander('Revenue estimate'):
         st.write(analyst_info['Revenue Estimate'])