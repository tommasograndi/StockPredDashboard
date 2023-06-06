# StockPredDashboard
 #### Financial dashboard for stock selection, analysis and price prediction.

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

 and wait for the browser to open a new page. <br> <br> <br>
 
 ### Dashboard previews and main functionalities
 
 ![Screen Recording 2023-06-06 at 16 36 49](https://github.com/tommasograndi/StockPredOptDashboard/assets/118896276/102fb96b-0d3b-46f9-ad90-4cecbfc968cc)

 
 
 

 _For more informations about the library Streamlit you can check the documentation at https://docs.streamlit.io_
