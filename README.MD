# Stock Prediction With R

This is an example of stock prediction with R using various technical indicators. R has various amazing libraries and I've tried to explain why I am using them in the course of this description. 

## Packages Required

xgboost, quantmod, highcharter, psych, pROC ; all downloadable from CRAN

### Prerequisites

Knowledge of R Programming

## Dataset Description

Data used in this project is obtained from Yahoo Finance using the built in `getSymbols()` call. This gives us data in the form of 

## Crime Density Map

Utilised the ggmap library to pull a roadmap from Google Maps. This was later overlaid with multiple grammar of graphics layers for density gradient and more.

![alt tag](https://user-images.githubusercontent.com/10093954/36330214-d1dc9344-1336-11e8-9b17-cb1a66cdea23.jpeg)

The map shows the crime density in Boston for a period of one year from February 2017 to February 2018. 

## Crime Statistics Districtwise

Barplot modeling using ggplot is done using the frequency of crimes committed.
A material theme is applied to make it look nice. First we generate a plot for the 16-17 year

![alt tag](https://user-images.githubusercontent.com/10093954/36330202-c209c126-1336-11e8-81ac-41a0234bc28b.jpeg)

Then we do the same for the 17-18 year.

![alt tag](https://user-images.githubusercontent.com/10093954/36330213-d1d155a6-1336-11e8-8536-bf77e99450e5.jpeg)

Now we can compare the percentage change between the two years and find out how much incidents of crime have changed

![alt tag](https://user-images.githubusercontent.com/10093954/36330215-d1e90052-1336-11e8-8052-07571a413c82.jpeg)

## Future

Will be working on a neural network to predict hotspots for the upcoming months. Will be interesting to see if that actually happens IRL.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

*  R Core Team (2013). R: A language and environment for statistical
  computing. R Foundation for Statistical Computing, Vienna, Austria.
  URL http://www.R-project.org/.

* R Packages used : Plyr, ggmap, ggplot2, lubridate

