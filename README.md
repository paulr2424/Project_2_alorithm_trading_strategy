## Algorithm Trading Strategy For The SPY ETF Using Supertrend Indicator + DeepAR Forecasting


### Summary
This project involved developing an algorithmic trading strategy to help automate the decision-making and execution of intraday trading against the SPY ETF (tracks S&P 500).  The solution leverages Alpaca Trading API (for historical data) and the SuperTrend indicator strategy to generate buy/sell signals for intraday trading and to automate order execution.  Furthermore, the Amazon DeepAR Forecasting algorithm was integrated as a real-time trading signal confirmation to use in concert with the automated buy/sell execution of the SuperTrend Trading Bot.

### Use of Supertrend Indicator
The choice for using this indicator was due in part to several reasons. Namely, we wanted an approach that was easy to understand and flexible for many timeframes.  Additionally, we liked that it provides precise buy and sell signals and works well with a trending market.  Since the scope of our project was time-constrained, we opted to implement a "Long-only" trading strategy.
#

### Model Training

#### SuperTrend Indicator
The Supertrend indicator requires no model training, only the implemetation of the Supertrend Indicator Formula as shown below in figure 1.

#### **Figure 1: Supertrend Indicator Formula**
![Supertrend Indicator Formula](/Images/supertrend-indicator-formula.png)
###### ©Copyright elearnmarkets


According to documentation, the indicator works very well in trending markets.  The buy signal is generated with the "Super Trend" closes below the most recent close price.  Similarly, the sell signal is generated when the "Super Trend" closes above the most recent close price.  Lastly, this trend indicator can be combined with other approaches to generate even "better" signals.  That is what we endeavored to find out here...

#### DeepAR Forecasting
The DeepAR Forecasting algorithm was selected as it is a supervised learning algorithm for forecasting scalar time series such as stock market closing prices.  This, theoretically, will allow for the model to extrapolate the time series data into the future for complementary trading signal confirmation.  Moreover,  DeepAR outperforms standard ARIMA and Exponential Smoothing techniques when datasets contain hundreds of related time series.  Figure 2 depicts is a visual overlay of train and test datasets for a given time series (1 day sample at 15 minute timeframe).

#### **Figure 2: Supertrend Indicator Formula**
![Supertrend Indicator Formula](/Images/overlay-train-test-time-series.png)
#

### Model Evaluation

The DeepAR Forecasting model uses various hyperparameters to allow for tuning and optimization of the neural network (RNN).  The settings for the parameters that were used for the analysis are shown below:
```
    epochs=40,
    num_cells=32,
    num_layers=3,
    dropout_rate=0.05,
    mini_batch_size=32,
    learning_rate=0.001,
    likelihood="gaussian",
    time_freq=frequency,
    early_stopping_patience=20,
    context_length=16,
    prediction_length=3
```

The DeepAR algorithm evaluates the trained model using the root mean squared error (RMSE) over the test data.  Figure 3 shows the formulat used for time series data:

#### **Figure 3: Root Mean Square Error (RMSE) Formula for Time Series Data **
![RMSE Formula for Time Series Data ](/Images/rmse-calculation.png)

Additionally, the algorithm evaluates the accuracy of the forecast distribution using weighted quantile loss.  Figure 4 shows the calculations for the weighted quantile loss:

#### **Figure 4: Weighted Quantile Loss Formula for Time Series Data **
![Weighted Quantile Loss Formula for Time Series Data](/Images/quantile-loss-calculation.png)
Below is the representative output from training of the model:
```
    [12/08/2021 21:27:18 INFO 140499396806016] #test_score (algo-1, RMSE): 2.1269036559074226
    [12/08/2021 21:27:18 INFO 140499396806016] #test_score (algo-1, mean_absolute_QuantileLoss): 83.20482245551214
    [12/08/2021 21:27:18 INFO 140499396806016] #test_score (algo-1, mean_wQuantileLoss): 0.0028445597400594325
    [12/08/2021 21:27:18 INFO 140499396806016] #test_score (algo-1, wQuantileLoss[0.1]): 0.0025912611382106652
    [12/08/2021 21:27:18 INFO 140499396806016] #test_score (algo-1, wQuantileLoss[0.2]): 0.003717977404373495
    [12/08/2021 21:27:18 INFO 140499396806016] #test_score (algo-1, wQuantileLoss[0.3]): 0.004196458815721761
    [12/08/2021 21:27:18 INFO 140499396806016] #test_score (algo-1, wQuantileLoss[0.4]): 0.004143057636348619
    [12/08/2021 21:27:18 INFO 140499396806016] #test_score (algo-1, wQuantileLoss[0.5]): 0.003426094011382414
    [12/08/2021 21:27:18 INFO 140499396806016] #test_score (algo-1, wQuantileLoss[0.6]): 0.0026046183189515726
    [12/08/2021 21:27:18 INFO 140499396806016] #test_score (algo-1, wQuantileLoss[0.7]): 0.0019821035437349818
    [12/08/2021 21:27:18 INFO 140499396806016] #test_score (algo-1, wQuantileLoss[0.8]): 0.0015710801414318299
    [12/08/2021 21:27:18 INFO 140499396806016] #test_score (algo-1, wQuantileLoss[0.9]): 0.0013683866503795571
    [12/08/2021 21:27:18 INFO 140499396806016] #quality_metric: host=algo-1, test RMSE <loss>=2.1269036559074226
    [12/08/2021 21:27:18 INFO 140499396806016] #quality_metric: host=algo-1, test mean_wQuantileLoss <loss>=0.0028445597400594325
```
#

### Results
From the live trading sessions, the Supertrend Indicator strategy worked well in identifying Buy/Sell signals at a 15-minute timeframe.  However, because the strategy was "long only", there was one day in our testing session where no trade signals were generated.  Overall, we identified that the DeepAR Model did not appear to effectively forecast future prices or our usage of the forecasted values was not appropriate as implemented.  Of particular note, the downtrend on 12/09 did not trigger and "buy" signals so feedback was limited.  Figure 5 shows the DeepAR Forecast model for end of day which include the closing price targets and corresponding mean predictions with 80% confidence interval for the next 3 successive closing prices.  In this case the predictions seem plausible, albeit the confidence interval is quite large and therefore may has questionable utility.

#### **Figure 5: Visualization of Target Prices w/Forecasted Mean and Confidence Interval for 12/09/2021 (SPY Downtrend) **
![Predictions 12/09/2021](/Images/predictions-12-09-2021.png)

On 12/10, there were several uptrend signals generated from the Supertrend Indicator.  However, the DeepAR Forecast was predicting a much lower mean.  Given that, and how we implemented the signal confirmation the end result was that the buy signals were repeatedly overriden for the entire day.

#### **Figure 6:  Visualization of Target Prices w/Forecasted Mean and Confidence Interval for 12/10/2021 (SPY Mixed W/Late Uptrend) **
![Predictions 12/10/2021](/Images/predictions-12-10-2021.png)

Figure 7 provides representative Trading Bot log output which captures the SuperTrend and DeepAR Model Forecast data along with the buy override message.

#### **Figure 7: Live Trading Logs (Sample) **
![Live Trading Logs](/Images/numerical-findings.png)
#

### Software Prerequisites
* Anaconda/Jupyter Notebook
* Python >= 3.7
* Git Bash

### Code Setup (clone Git Repository)
* open a git bash terminal on your local computer
* type `git clone https://github.com/gakees/project_2_algorithm_trading_strategy.git`
* then type `cd project_2_algorithm_trading_strategy/Code/`
* create a `.env` file and add the following content:
```
    ALPACA_API_KEY=[your alpaca api key goes here...]
    ALPACA_SECRET_KEY=[your alpaca secret key goes here...]

    AWS_REGION_NAME=[your hosted region name goes here...]
    AWS_ENDPOINT_NAME=[your sagemaker hosted endpoint name goes here...]
```

* To run the [supertrend_long_trading_strategy_bot.ipynb](/Code/supertrend_long_trading_strategy_bot.ipynb) notebook you will need to...
    * activate the conda `algotrading` environment and then install boto3, python-dotenv and alpaca-trade-api
    * type `conda activate algotrading`
    * type `pip install boto3[crt]`
    * type `pip install python-dotenv`
    * type `pip install alpaca-trade-api`
* To run the [stock_forecasting_model_setup.ipynb](/Code/stock_forecasting_model_setup.ipynb) notebook, see the next section below
    - NOTE: When running the SageMaker notebook in AWS, you will need to also upload the `.env` file into the /Code folder

## DeepAR Model training and endpoint deployment (need help using AWS Services? [Look here](https://aws.amazon.com/sagemaker/).)
* login to AWS SageMaker Studio with a suitable role (IAM Access Role preferred)
* Create a new SageMaker Notebook instance and S3 Bucket in the same AWS Region (NOTE: important that they be deployed to same region)
* Within the SageMaker notebook clone the git repository to access the `stock_forecasting_model_setup.ipynb` notebook
* Run the notebook
    * NOTE: this will take some time for the model training and endpoint deployment
    * When the run is completed the SageMaker endpoint will now be available
    * Go back to the SageMaker Dashboard
        * Access the Inference Menu
            * Select the Endpoints Menu Item
            * This will show a list of Endpoints deployed
            * Make note of the endpoint name and update the `.env` file (AWS_ENDPOINT_NAME=[your sagemaker hosted endpoint name goes here...])

### How to launch the Supertrend (Long) Trading Bot
* from within the cloned git repository
    * type `jupyter lab` (from git bash)
    * then open the `supertrend_long_trading_strategy_bot.ipynb` file within the browser (/Code/supertrend_long_trading_strategy_bot.ipynb)
