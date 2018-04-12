"Term Project For IE 7275: Data Mining for Engineers"
library(xgboost)
library(quantmod)
library(TTR)
stocksym <-c("XLF","JPM","IYF","FNCL")
#Get stock data using getSymbols call
getSymbols(stocksym)
#Dataframe for JP Morgan stock for last 5 years
JPM<-last(JPM,"5 years")
#Similarly for XLF, IYF and FNCL
XLF<-last(XLF,"5 years")
IYF<-last(IYF,"5 years")
FNCL<-last(FNCL,"5 years")
#Lets Visualise the data
library(highcharter)
#Highcharter charts are look really good on the eyes
highchart(type = "stock") %>% hc_add_series(JPM,type="ohlc") %>% 
  hc_add_series(XLF,type="ohlc") %>% hc_add_series(FNCL, type="ohlc") %>%
  hc_add_series(IYF, type="ohlc") %>% hc_legend(enabled=TRUE) %>% hc_title(text="OHLC Prices For The Last 5 Years")
#If you would like candlecharts, susbstitute type="line" with type="ohlc"
#Its not a good idea to build a model based on the prices themselves.
#Time series data tends to be correlated in time and in turn exhibits significant autocorrelation
#Learnt that from this article here https://www.linkedin.com/pulse/how-use-machine-learning-time-series-forecasting-vegard-flovik-phd/
#Bottom line, we need to use stationary data to build a model -> Feature Engineering
#Let's define a few technical indicators to build a model
#Relative Strength Indicator
rsiFNCL <-RSI(FNCL$FNCL.Close, n=14, maType = "WMA")
plot(rsiFNCL, type = 'l')
rsiIYF <-RSI(IYF$IYF.Close, n=14, maType = "WMA")
plot(rsiIYF, type = 'l')
rsiXLF<-RSI(XLF$XLF.Close, n=14, maType = "WMA")
plot(rsiXLF, type = 'l')
#Average Directional Indicator
adxFNCL <- data.frame(ADX(FNCL[,c("FNCL.High","FNCL.Low","FNCL.Close")]))
plot(adxFNCL$ADX,type = 'l')
adxIYF <- data.frame(ADX(IYF[,c("IYF.High","IYF.Low","IYF.Close")]))
plot(adxIYF$ADX,type = 'l')
adxXLF <- data.frame(ADX(XLF[,c("XLF.High","XLF.Low","XLF.Close")]))
plot(adxXLF$ADX,type = 'l')
#Parabolic Stop and Reversal Indicator
sarFNCL <- SAR(FNCL[,c("FNCL.High","FNCL.Low")], accel = c(0.02, 0.2))
trendfncl=FNCL$FNCL.Close-sarFNCL
plot(trendfncl)
sarIYF <- SAR(IYF[,c("IYF.High","IYF.Low")], accel = c(0.02, 0.2))
trendiyf=IYF$IYF.Close-sarIYF
plot(trendiyf)
sarXLF <- SAR(XLF[,c("XLF.High","XLF.Low")], accel = c(0.02, 0.2))
trendxlf=XLF$XLF.Close-sarXLF
plot(trendxlf)
#Induce a lag to avoid look ahead bias
rsiFNCL=c(NA,head(rsiFNCL,-1))
rsiIYF=c(NA,head(rsiIYF,-1))
rsiXLF=c(NA,head(rsiXLF,-1))
adxFNCL$ADX=c(NA,head(adxFNCL$ADX,-1))
adxIYF$ADX=c(NA,head(adxIYF$ADX,-1))
adxXLF$ADX=c(NA,head(adxXLF$ADX,-1))
trendfncl=c(NA,head(trendfncl,-1))
trendiyf=c(NA,head(trendiyf,-1))
trendxlf=c(NA,head(trendxlf,-1))
#Objective is to predict up or down on the daily price
#A clear binary classification problem
#Create the response variable
price=JPM$JPM.Close-JPM$JPM.Open
price=c(NA,head(price,-1))
class=ifelse(price > 0,1,0)
#Combine input features into a matrix
model_df = data.frame(class,rsiFNCL,rsiIYF,rsiXLF,adxFNCL$ADX,adxIYF$ADX,adxXLF$ADX,trendfncl,trendiyf,trendxlf)
model=data.matrix(model_df)
library(psych)
fa.parallel(model[,-1], fa ="pc", n.iter = 100, show.legend = FALSE, main = "Scree Plot With Parallel Analysis")
princicomp <- principal(model[,-1], nfactors = 2)
princicomp
model=na.omit(model)
colnames(model)=c("class","rsiFNCL","rsiIYF","rsiXLF","adxFNCL","adxIYF","adxXLF","trendfncl","trendiyf","trendxlf")
#Data prep for training and test
train_sz=2/3
breakpt = nrow(model)*train_sz
traindata=model[1:breakpt,]
testdata=model[(breakpt+1):nrow(model),]
#Break train data into X and Y- Response Variable
X_train=traindata[,2:10]
Y_train=traindata[,1]
class(X_train)[1]
class(Y_train)
#Do the same for test data
X_test=testdata[,2:10]
Y_test=testdata[,1]
#Time to start XGBoost
set.seed(3213)
dtrain=xgb.DMatrix(data = X_train, label=Y_train)
xgmodel=xgboost(data=dtrain, nround=10, objective="binary:logistic")
set.seed(27)
model_xgb_pca <-train(outcome ~ .,
                      data=val_train_data,
                      method="xgbTree",
                      preProcess = "pca",
                      trControl = trainControl(method = "repeatedcv", number = 5, repeats = 10, verboseIter = FALSE))
#Use cross validation
dtrain=xgb.DMatrix(data = X_train, label=Y_train)
cv=xgb.cv(data = dtrain, nround=10, nfold = 5, objective="binary:logistic")
preds=predict(xgmodel,X_test)
prediction=as.numeric(preds>0.543)
#Find AUC for this
library(pROC)
auc<-auc(Y_test,prediction)
print(paste('AUC Score:',auc))
rocg<-roc(Y_test,prediction)
plot(rocg, col="blue")
tablesamp <- table(prediction, Y_test)
tablesamp
library(class)
#Let's run another classification algorithm to test if our data is valid.
#This is a small script I wrote to automate and test for accuracy for up to
#k cases of KNN classification.
"Enter number of cases you wanna test for training data, 
testdata, trainresponse and testresponse"
#Testing for accuracy which is TP + TN/ Total No. of Test Cases
getknnerr <-function(n, traindata, testdata, trainresp, testresp) {
  
  ncount=0
  err=0
  errcount=0
  tablen=0
  knnsamp <- knn(traindata, testdata, trainresp, k=1)
  tablesamp <- table(knnsamp, testresp)
  print(tablesamp)
  err=(tablesamp[1,1]+tablesamp[2,2])/(nrow(testdata))
  cat("Base error at k=1 :",err)
  for(i in 2:n){
    knnsamp2 <- knn(traindata, testdata, trainresp, k=i)
    tablesamp2 <- table(knnsamp2, testresp)
    errcheck=(tablesamp2[1,1]+tablesamp2[2,2])/(nrow(testdata))
    if(errcheck>err){
      ncount=i
      err=errcheck
      tablen=tablesamp2
      }
  }
  cat("\nfinal accuracy score:",err)
  cat("\nachieved at k=",ncount)
  tablen
  return(ncount)
}
#The function returns the k value with the highest accuracy.
knn<-getknnerr(10, traindata = X_train, testdata = X_test,trainresp = Y_train,testresp = Y_test)
knnreturn <- knn(X_train, X_test, Y_train, k=knn)
tablesamp <- table(knnreturn, Y_test)
tablesamp
knnreturn <- sapply(knnreturn, as.numeric)
#Plot roc curve
rocg <- roc(Y_test,knnreturn)
plot(rocg)
aucsc <- auc(Y_test,knnreturn)
aucsc
#This library helps us visualise the xgboost trees
library(DiagrammeR)
xgb.plot.tree(model = xgmodel)
# View only the first tree in the XGBoost model
xgb.plot.tree(model = xgmodel, n_first_tree = 1)
