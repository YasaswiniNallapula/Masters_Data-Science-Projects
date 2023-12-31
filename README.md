# Masters_Data-Science-Projects
---
title: "Prediction of customers of bank who would subscribe to a term deposit with Data Mining Algorithms"
author: "Yasaswini"
date: "2023-03-05"
output: "html_document"
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## R Markdown

This is an R Markdown document. Markdown is a simple formatting syntax for authoring HTML, PDF, and MS Word documents. For more details on using R Markdown see <http://rmarkdown.rstudio.com>.

When you click the **Knit** button a document will be generated that includes both content as well as the output of any embedded R code chunks within the document. You can embed an R code chunk like this:


## 1 Introduction

### 1.1 Background
Marketing new potential customers and retaining them over the long term is a constant challenge for banks. To reach profitable customers, banks often use media such as social and digital media, customer service and strategic partnerships. But is it possible for banks to market to specific locations, communities, and groups of people? Fortunately, with the advent of machine learning technology, banking institutions are leveraging data and analytics solutions to target specific target customers and to predict which customers accurately and intelligently are likely to purchase financial products and services. 
Using Banking marketing dataset, I would like to perform exploratory data analysis, statistical analysis if required, and then use Logistic Regression to build machine learning model
The data is related with direct marketing campaigns of a banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.


### 1.2 Data Description

Dataset Source:
https://datahub.io/machine-learning/bank-marketing#resource-bank-marketing. 

Input variables—
# bank client data:  
1	age	numeric  
2	job	type of job (categorical: ‘admin’, ’blue-collar’, ’entrepreneur’, ‘housemaid’, ‘management’, ‘retired’, ‘self-employed’, ‘services’, ‘student’, ‘technician’, ‘unemployed’, ‘unknown’)  
3	marital	marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)  
4	education	(categorical: ‘unknown’,’secondary’,’primary’,’tertiary’)  
5	default	has credit in default? (categorical: 'no','yes')  
6	balance	average yearly balance, in euros (numeric)  
7	housing	has housing loan? (categorical: 'no','yes','unknown')  
# related with the last contact of the current campaign:  
8	loan	has personal loan? (categorical: 'no','yes','unknown')  
9	contact	contact communication type (categorical: 'cellular','telephone', ‘unknown’)  
10	day	last contact day of the month (numeric)  
11	month	last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')  
# other attributes:  
12	duration	last contact duration, in seconds (numeric)  
13	campaign	number of contacts performed during this campaign and for this client (numeric, includes last contact)  
14	pdays	number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)  
15	previous	number of contacts performed before this campaign and for this client (numeric)  
16	Poutcome	outcome of the previous marketing campaign (categorical: 'failure',unknown,'success', ‘other’)  
#target/output variable  
17	y	has the client subscribed a term deposit? (numerical: 1, 2)



### 1.3 Problem Statement

A term deposit is a fixed term deposit of money in an account of a financial institution. If a client or investor decides to deposit or invest in any of these accounts, they agree not to withdraw money for a period of time (from 1 month to 30 years) in exchange for a higher amount. interest rate on that account. Banks can use this money to invest elsewhere or lend it to someone else for an agreed period of time. In other words, fixed deposits are guaranteed to receive money at a fixed interest rate for a certain period of time. As term deposits are an important source of income for banks, banks invest large sums of money and focus on marketing campaigns to attract more customers to term deposits.

## 2 Data Exploration

### 2.1 Load data

```{r}
# Load packages
#installing libraries
library(caret)
library(caTools)
#install.packages("DataExplorer")
library(DataExplorer)
#install.packages("ROCR")
library(ROCR)
#install.packages("dplyr")
library(dplyr)
library(glue)
library(e1071)
#install.packages("randomForest")
library(randomForest)
#install.packages("naivebayes")
#library(naivebayes)
```

Now that our packages are loaded, let’s read in and take a look at the data (Data validation, Data Cleaning, Data Pre-processing)

```{r}
#Load the dataset into R
bank <- read.csv("path_to_file",header=TRUE,sep=",")
colnames(bank) = c("age", "job", "marital", "education", "default", "balance", "housing", "loan", 
                   "contact", "day", "month", "duration", "campaign", "pdays", "previous", 
                   "poutcome", "y")

#description of dataset
summary(bank)
str(bank)
dim(bank)


#Data Validation
##check for duplicate rows
sum(duplicated(bank))
##check for Missing values
sapply(bank, function(x) sum(is.na(x)))
plot_missing(bank, title='plot for missing values', ggtheme=theme_linedraw(),
             theme_config=list(legend.position=c("bottom")))

#convert character to factor type
bank <- as.data.frame(unclass(bank), stringsAsFactors = TRUE)
bank_lr <- bank
#convert int to factor type
bank$y <- factor(bank$y, levels = c(1,2), labels = c('no', 'yes'))
bank_lr$y <- ifelse(bank_lr$y==2, 1,0)
#print all the levels of factor variables to find any missing values
levels(bank$job)
levels(bank$marital)
levels(bank$education)
levels(bank$default)
levels(bank$housing)
levels(bank$loan)
levels(bank$contact)
levels(bank$month)
levels(bank$poutcome)
levels(bank$y)

```


### 2.2 Below are the R commands for running the EDA

Here, I will try to use visualizations to understand the data, to understand the correlations between variables (univariate analysis, multivariate analysis).
```{r}
#data analysis
table(bank$y)
round(prop.table(table(bank$y)) * 100, digits = 1)


job_distribution <- bank %>% group_by(job) %>% summarise(job_count = n()) %>% arrange(-job_count)
job_distribution_plot <- ggplot(data = job_distribution, aes(x = job_count, 
                                                             y = reorder(job, job_count), 
                                                             text = glue("No. of customers: {job_count}")
)) +
  geom_col(aes(fill = job)) +
  labs( title = "Job Distribution of Customers",
        x = "No. of Jobs",
        y = "jobs"
  ) +
  theme_minimal() +
  theme(legend.position = "none")
job_distribution_plot

education_distribution <- bank %>% group_by(education) %>% summarise(ed_count = n())
education_distribution_plot <- ggplot(data = education_distribution, aes(y = ed_count, 
                                                                         x = education, 
                                                                         text = glue("No. of customers: {ed_count}")
)) +
  geom_col(aes(fill = "brickred")) +
  labs( title = "Education Distribution of Customers",
        x = "Education Level",
        y = ""
  ) +
  theme_minimal() +
  theme(legend.position = "none")
education_distribution_plot


mu <- bank %>% group_by(y) %>% summarize(grp.mean=mean(age))
ggplot (bank, aes(x=age)) + 
  geom_histogram(color = "blue", fill = "blue", binwidth = 5) +
  facet_grid(cols=vars(y)) + 
  ggtitle('Age Distribution by Subscription') + ylab('Count') + xlab('Age') +
  scale_x_continuous(breaks = seq(0,100,5)) +
  geom_vline(data=mu, aes(xintercept=grp.mean), color="red", linetype="dashed")

ggplot(data = bank, aes(x=education, fill=y)) +
  geom_bar() +
  ggtitle("Term Deposit Subscription based on Education Level") +
  xlab(" Education Level") +
  guides(fill=guide_legend(title="Subscription of Term Deposit"))

ggplot(data = bank, aes(x=job, fill=y)) +
  geom_bar() +
  ggtitle("Term Deposit Subscription based on job") +
  xlab(" Job") +
  guides(fill=guide_legend(title="Subscription of Term Deposit"))

```

### 2.3 Data Preparation for model building
Transforming the numeric variables using scale (Z-score standardization) to handle the distance calculation. Splitting the dataset into training and test data for the purpose of building Data Mining techniques which will use training data and then they will make predictions on the testing data.

```{r}
#transforming the numerical variables using scale
bank_lr[c(1,6,10,12,13)] <- scale(bank_lr[c(1,6,10,12,13)])
#split the dataset into training and testing for logistic regression
set.seed(112)
split = sample.split(bank_lr$y,SplitRatio = 0.70)
bank_lr_training = subset(bank_lr, split == TRUE)
bank_lr_test = subset(bank_lr, split == FALSE)

# Split the data into training and testing sets for the Naive Bayes model
set.seed(1234)  # for reproducibility
trainIndex <- createDataPartition(bank$y, p = 0.7, list = FALSE)
bank_nb_train <- bank[trainIndex, ]
bank_nb_test <- bank[-trainIndex, ]

#split the data into training and testing for the Random Forest
bank_rf <- subset(bank, select = -c(duration))
set.seed(42) # Set random seed for reproducibility
train_indices_rf <- createDataPartition(bank_rf$y, p = 0.7, list = FALSE)
bank_rf_train <- bank_rf[train_indices_rf, ]
bank_rf_test <- bank_rf[-train_indices_rf, ]

```


## 3 Model Building:

### 3.1 Logisitc Regression Model:


```{r}

#logistic regression model

binclass_eval = function (actual, predict) {
  cm = table(as.integer(actual), as.integer(predict), dnn=c('Actual','Predicted'))
  ac = (cm['1','1']+cm['0','0'])/(cm['0','1'] + cm['1','0'] + cm['1','1'] + cm['0','0'])
  pr = cm['1','1']/(cm['0','1'] + cm['1','1'])
  rc = cm['1','1']/(cm['1','0'] + cm['1','1'])
  fs = 2* pr*rc/(pr+rc)
  list(cm=cm, recall=rc, precision=pr, fscore=fs, accuracy=ac)
}


plot_pred_type_distribution <- function(df, threshold) {
  v <- rep(NA, nrow(df))
  v <- ifelse(df$pred >= threshold & df$y == 1, "TP", v)
  v <- ifelse(df$pred >= threshold & df$y == 0, "FP", v)
  v <- ifelse(df$pred < threshold & df$y == 1, "FN", v)
  v <- ifelse(df$pred < threshold & df$y == 0, "TN", v)
  
  df$pred_type <- v
  
  ggplot(data=df, aes(x=y, y=pred)) + 
    geom_violin(fill='black', color=NA) + 
    geom_jitter(aes(color=pred_type), alpha=0.6) +
    geom_hline(yintercept=threshold, color="red", alpha=0.6) +
    scale_color_discrete(name = "type") +
    labs(title=sprintf("Threshold at %.2f", threshold))
}

#creating the LR classifer
classifier.lm = glm(formula = y ~ .,
                    family = binomial,
                    data = bank_lr_training)

#Evaluating the LR model
pred_lm = predict(classifier.lm, type='response', newdata=bank_lr_test[-17])
predictions_LR <- data.frame(y = bank_lr_test$y, pred = NA)
predictions_LR$pred <- pred_lm
plot_pred_type_distribution(predictions_LR,0.30)

test.eval.LR = binclass_eval(bank_lr_test[, 17], pred_lm > 0.30)
test.eval.LR$cm
acc_LR=test.eval.LR$accuracy
prc_LR=test.eval.LR$precision
rc_LR=test.eval.LR$recall
cat("Accuracy:  ",   acc_LR,
    "\nPrecision: ", prc_LR,
    "\nRecall: ",rc_LR)


rocr.pred.lr = prediction(predictions = pred_lm, labels = bank_lr_test$y)
rocr.perf.lr = performance(rocr.pred.lr, measure = "tpr", x.measure = "fpr")
rocr.auc.lr = as.numeric(performance(rocr.pred.lr, "auc")@y.values)

rocr.auc.lr

plot(rocr.perf.lr,
     lwd = 3, colorize = TRUE,
     print.cutoffs.at = seq(0, 1, by = 0.1),
     text.adj = c(-0.2, 1.7),
     main = 'ROC Curve')
mtext(paste('Logistic Regression - auc : ', round(rocr.auc.lr, 5)))
abline(0, 1, col = "red", lty = 2)

```

### 3.2  Naive Bayes Model


```{r}

# Train the Naive Bayes model
nb_model <- naiveBayes(y ~ ., data = bank_nb_train)
nb_model

# Evaluate the model on the testing set
predictions <- predict(nb_model, newdata = bank_nb_test,na.action = na.pass)
confusionMatrix(predictions, bank_nb_test$y)

```


### 3.3  Random Forest Model


```{r}

# Creating a random forest classifier with 100 trees
rf_model <- randomForest(y ~ ., data = bank_rf_train, ntree = 100)
rf_model

# Making predictions on the testing data
y_pred <- predict(rf_model, bank_rf_test)

# Evaluating the accuracy of the model
accuracy <- confusionMatrix(y_pred, bank_rf_test$y)$overall["Accuracy"]
print(paste("Accuracy:", round(accuracy, 4)))

```



## 4 Conclusion

Accuracy for different models:

The Logistic Regression model has a high accuracy of 90.23%, but the sensitivity/recall value is very small at 54.6%. Recall measures how good our model is at correctly predicting positive classes.
The Naive bayes model has an accuracy of 88%. The Random Forest model has an accuracy of 89.12%





