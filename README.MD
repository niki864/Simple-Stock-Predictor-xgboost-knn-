# Stock Prediction With R

This is an example of stock prediction with R using ETFs of which the stock is a composite. To get rid of seasonality in the data, we used technical indicators like RSI, ADX and Parabolic SAR that more or less showed stationarity. The goal of the project is to predict if the stock price today will go higher or lower than yesterday. This work was done as a term project for the course IE 7275: Data Mining for Engineers @ Northeastern University.

## Packages Required

* xgboost
* quantmod
* highcharter
* psych
* pROC 

All downloadable from CRAN repositories

### Prerequisites

* Knowledge of R Programming
* R Studio

## Dataset Description

Data used in this project is obtained from Yahoo Finance API using *quantmod* built in function `getSymbols()`. This gives us data in the form of time series xts objects. Using the `last()` function we can specify our time range. I'm using the last 5 years of data for this project.
The following stocks/ETFs were used:
* Response Variables: JPMorgan - Open, Close 
* Predictor Variables: FNCL - Fidelity MSCI Financials Index, IYF - iShares US Financials ETF, XLF - Financial Select Sector SPDR Fund

A keen observer would note that all the 3 Predictor variables are ETFs that relate to banking and finance stocks. JPM is composite of all the three above funds.

## Visualisation of Price History

The __highcharter__ library is a brilliant tool for generating visually appeasing and interactive charts. Although it's free for non-commercial/academic use, it requires a license for commercial use though. This is the first time I'm playing with this library and I gotta say, it's really neat.

The following chart was generated using highcharter.

<img src="https://user-images.githubusercontent.com/10093954/38693940-69c9763c-3e56-11e8-809c-bc00ef322693.jpeg" alt="Drawing" style="width:100%;"/>

Download the chart [here](chart.html)


## Prediction Model Description


Our goal in this project is to use ETFs to predict the value of one composite stock. The premise for this is that, we can think of an ETF as a representative for the entire industry. Banking and financial firms are all pretty much correlated to each other as even a minor policy change could potentially affect all of them. Thus, by using the performance of the ETF to train our Machine Learning models, we can arrive at a healthy and reasonable prediction for target stock : __JP Morgan(JPM)__

__Note:__ _This a stock prediction project done as part of a term assignment and clearly, is not to be taken as sound investment advice. Predicting stock prices in the market is more challenging and requires enormous effort and way more degrees and qualifications than what we currently have :) Cheers!

One common mistake in using time series data is that the data tends to exhibit seasonality and to arrive at an accurate measure, we need to convert it into a stationary data. Check out this article by Vegard Flovik where he talks more about this [https://www.linkedin.com/pulse/how-use-machine-learning-time-series-forecasting-vegard-flovik-phd/](https://www.linkedin.com/pulse/how-use-machine-learning-time-series-forecasting-vegard-flovik-phd/)

One way we can go about doing this is differencing the data. But since this is financial data, the _quantmod_ package has a lot technical indicator functions which we can use to generate indicator data that more or less gets rid of seasonality.

Some of the indicators, we have used are:

RSI - Relative Stregth Index (A measure of how the stock performed scaled to 0-100 w.r.t the Weighted Moving Average)

<img src="https://user-images.githubusercontent.com/10093954/38693864-2e6c6ec8-3e56-11e8-8cfa-2ee493b11bf8.jpeg" alt="Drawing" style="width:100%;"/>

ADX - Average Directional Indicator

<img src="https://user-images.githubusercontent.com/10093954/38693863-2e5a0e36-3e56-11e8-9631-022569cf1f49.jpeg" alt="Drawing" style="width:100%;"/>

Parabolic SAR Trend- Stop and Reverse Indicator

<img src="https://user-images.githubusercontent.com/10093954/38693862-2e4c9d50-3e56-11e8-8e17-0b62341996a6.jpeg" alt="Drawing" style="width:100%;"/>

After munging out all the numbers for the indicators, we then feed it into our model. We also incorporate a lag of 1 day to avoid a lookahead bias on the data.

## Machine Learning Algorithm

We will be using the xgboost algorithm with the goal of binary logistic regression. After data preparation into training (_approx 70%_ )and test (_approx 30%_) sets, we then feed it to the algorithm. 

Here's the ROC Curve for our first run on 10 rounds. 

<img src="https://user-images.githubusercontent.com/10093954/38693861-2e3d66dc-3e56-11e8-8c44-0ddb7f15f0ed.jpeg" alt="Drawing" style="width:100%;"/>

We achieved an AUC of :  0.591939755047997

To verify this claim and to further test our model, we ran KNN classification on the data set. 
Using a handy script I wrote, we arrived at a optimum K value of 8.

This is the ROC Curve for the k=8 KNN Classification

<img src="https://user-images.githubusercontent.com/10093954/38693860-2e2ae28c-3e56-11e8-9b5a-52971cd871ed.jpeg" alt="Drawing" style="width:100%;"/>

We achieved an AUC of : 0.5728


## XGBoost Visualisation

The DiagrammeR R package allows us to visualise the tree structure generated by xgboost. Here's the entire structure.

<img src="https://user-images.githubusercontent.com/10093954/38693859-2e1b25ea-3e56-11e8-839a-d396bef199c8.jpeg" alt="Drawing" style="width:100%;"/>

IMO, it looks really cool.


This is what we get when we zoom into one tree
<img src="https://user-images.githubusercontent.com/10093954/38693858-2e0d0294-3e56-11e8-87a0-8a1c67e0b02b.jpeg" alt="Drawing" style="width:100%;"/>

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

*  R Core Team (2013). R: A language and environment for statistical
  computing. R Foundation for Statistical Computing, Vienna, Austria.
  URL http://www.R-project.org/.

* R Packages used : xgboost, quantmod, highcharter, psych, pROC 

