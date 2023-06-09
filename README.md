## Financial dashboard for stock selection, analysis and price prediction.

 ### Structure of the repository
 For the project of the second module of IT coding I chose the Stock Price Prediction, but I also decided to extend this prediction and implement it into a web dashboard that could allow the user to select a stock, visualize informations and request a prediction based on two main models: **ARIMA** and **XGBOOST**. 

 For this reason the code consists of two parts: 
 1. The notebook called `report.ipynb` represents an in depth explanation and thougt process of the stages of data gathering, processing, analysis and the creation of the predicting techniques. In the notebook there is also the explanation of the functions created to display the data and the outcome of the predictions inside a web app DashBoard created through the Streamlit Library. 
 2. Additionally there is a script called `dashboard.py` that contains synthetically the most important parts presented in the notebook, the functions that are used to manipulate the data and the whole interface of the dashboard. This script is the one to be runned by streamlit in order to access the dashboard and interact with it. 

 ### Access the dashboard. 
 In order to access the dashboard, is fundamental first to install all the required libraries. See requirements for details.
 After this, you'll have to activate the Python environment (in my case is called `ITproject`) with the necessary libraries installed. I used Anaconda to create and manage it and I can access it from the terminal with:
 
    conda activate ITproject

 Secondly, you have to save the script in a local folder and locate from the terminal into the same directory ('foldername') of the script with the command 
 
    cd foldername

 You can also clone this repository and access it from the relative GitHub folder. 

 Finally, you can run:

    streamlit run dashboard.py

 and wait for the browser to open a new page. 
 
 ### Dashboard previews and main functionalities
 
 <img width="1560" alt="Screenshot 2023-06-09 at 16 23 27" src="https://github.com/tommasograndi/StockPredDashboard/assets/118896276/0cf70f0d-6c00-4ea5-aade-217c309fbcf7">
 <br> <br>
 
 The dashboard is organized in the following way: 

1. **User input** <br>
First, we have two selectboxes and one textbox in which we can choose between 3 markets () and pick a stock in the list of stocks traded in the selected market (we can also write the name of the stock and search for it). <br>
After choosing the stock, on the right side there is an input textbox in which we can put the initial date from which we want to download the data.<br>
Every time the user interact with this upper boxes will change the inpput parameters for all the functions that download the data to be fed to the rest of the script and to the following functions that display the informations and calculate the predictions. 

<img width="1589" alt="image" src="https://github.com/tommasograndi/StockPredOptDashboard/assets/118896276/663e4b2d-2915-468d-a5b9-da1de3706745">
<br> 

2. **Content output**  <br>
After the user has interacted with the input boxes, the script will execute the functions and will display the following informations:

   - *Price chart*. This tab will show the Company stock price plot and below the downloaded data. The user can interact with the chart thanks to the `plotly` renderer. 
      ![Screen Recording 2023-06-09 at 15 18 33](https://github.com/tommasograndi/StockPredOptDashboard/assets/118896276/fe1e562e-b3c2-47ad-bf85-36f17c68ac33)
   <br>

   - *Technical analysis*. This tab will display the original stock price with some technical indicators plotted along with it. The three technical indicators are Simple Moving Average, Exponential Moving Average and RSI (that is plotted in a separate figure).
     <img width="1595" alt="image" src="https://github.com/tommasograndi/StockPredOptDashboard/assets/118896276/4fcc5632-706b-4979-93ee-b200963e353a">
   <br>

   - *Prediction*. This is the most important tab of the dashboard, the one with the two models studied and created according to the process explained in the `report.ipynb` notebook.  As we know, the two models are ARIMA and XGBOOST. <br>
   *ARIMA:*
   <img width="1548" alt="image" src="https://github.com/tommasograndi/StockPredDashboard/assets/118896276/ee6442b1-a71d-4b21-a9c3-59f8e3e480f6">
   <br>
   
   *XGBOOST:*
   <img width="1547" alt="image" src="https://github.com/tommasograndi/StockPredDashboard/assets/118896276/83082536-9ddb-4f6c-b94c-2ae28fc32b6c">

   <br>

   - *Statistics*. This tab simply display some statistics of the stock scrape with `yahoo_fin` package from Yahoo Finance. 
     <img width="496" alt="image" src="https://github.com/tommasograndi/StockPredDashboard/assets/118896276/2da50081-c51f-4ae5-bb10-45260d87115c">
     <br>

   - *Analysts' estimates*. As the previous tab, this one is printing analysts predictions and forecast for the company main figures. The informations are obtained through a `yahoo_fin` function. 
     <img width="679" alt="image" src="https://github.com/tommasograndi/StockPredDashboard/assets/118896276/45c85a8b-8bdc-4445-b5d3-20e28818e470">

   


<br> <br> <br>
 _For more informations about the library Streamlit you can check the documentation at https://docs.streamlit.io_
