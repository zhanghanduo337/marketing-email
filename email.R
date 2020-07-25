setwd('/Users/zhanghanduo/Desktop/data/email')
rm(list = ls())
library(dplyr)
library(ggplot2)
library(PRROC)
library(tidyverse)
library(plotly)
library(gtable)
library(randomForest)
library(gbm)
library(caret)

################################################################### import data

email_open = read.csv('email_opened_table.csv')
email = read.csv('email_table.csv')
link_open  = read.csv('link_clicked_table.csv')

################################################################### EDA

sum(unique(email_open$email_id) %in% unique(email$email_id) == F)
sum(unique(link_open$email_id) %in% unique(email$email_id) == F)
sum(unique(link_open$email_id) %in% unique(email_open$email_id) == F) 
#50 link_open but not email_open

length(unique(email_open$email_id))/length(unique(email$email_id))#10.345%
length(unique(link_open$email_id))/length(unique(email$email_id))#2.119%

email$email_open = if_else(email$email_id %in% unique(email_open$email_id),1,0)
email$link_open = if_else(email$email_id %in% unique(link_open$email_id),1,0)

df = email
summary(df)
attach(df)
par(mfrow = c(3,2))
link_text_version = df %>% 
  group_by(email_text,email_version) %>%
  summarise(avg_ctr = mean(link_open)) 
link_text_version
barplot(link_text_version$avg_ctr,xlab = 'text&version',
        ylab = 'avg_CTR',main = 'CTR_text_version',
        names.arg = c('long_generic','long_personalized',
                      'short_generic','short_personalized'))


link_purchase = df %>% 
  group_by(user_past_purchases) %>%
  summarise(avg_ctr = mean(link_open)) 
link_purchase
barplot(link_purchase$avg_ctr,xlab = 'past_purchase',
        ylab = 'avg_CTR',main = 'CTR_purchase',
        names.arg = c(1:23))


link_hour = df %>% 
  group_by(hour) %>%
  summarise(avg_ctr = mean(link_open))
link_hour 
barplot(link_hour$avg_ctr,xlab = 'hour',
     ylab = 'avg_CTR',main = 'CTR_hour',names.arg = c(1:24))


link_wk = df %>% 
  group_by(weekday) %>%
  summarise(avg_ctr = mean(link_open)) 
link_wk$weekday = factor(link_wk$weekday, levels= c("Sunday", "Monday", 
                                   "Tuesday", "Wednesday", "Thursday", "Friday", 
                                   "Saturday"))
link_wk = link_wk[order(link_wk$weekday), ]
barplot(link_wk$avg_ctr,xlab = 'weekday',
     ylab = 'avg_CTR',main = 'CTR_wk',names.arg = 
       c("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"))


link_country = df %>% 
  group_by(user_country) %>%
  summarise(avg_ctr = mean(link_open))
link_country
barplot(link_country$avg_ctr,xlab = 'country',
        ylab = 'avg_CTR',main = 'CTR_country',names.arg = 
          c("UK", "US", "ES", "FR"))

################################################################### Random Forest

temp = df[,-c(1,8)]

set.seed(1)
train = sample(1:nrow(df),nrow(df)*2/3)

temp$link_open = as.numeric(as.character(temp$link_open))
temp$email_text = as.factor(temp$email_text)
temp$email_version = as.factor(temp$email_version)
temp$weekday = as.factor(temp$weekday)
temp$user_country = as.factor(temp$user_country)
temp$user_past_purchases = as.numeric(as.character(temp$user_past_purchases))
temp$hour= as.factor(temp$hour)
rf.link = randomForest(link_open~.,data = temp,subset = train,
                       mtry = 4,importance = TRUE)
rf.link$confusion#class error too high
link_pred_rf = predict(rf.link,temp[-train,])

varImpPlot(rf.link) 

# cutoff point - 1.05 is to add additional margin
cutoff  = 1.05*mean(temp$link_open)
# get the outcome of predictions - sanity check
rf_pred = rep(0,nrow(temp[-train,]))
rf_pred[link_pred_rf>=cutoff]=1
table(rf_pred,temp[-train,]$link_open)
mean(rf_pred==temp[-train,]$link_open)
test.err.rf = mean((link_pred_rf-temp[-train,]$link_open)^2)

rf_scores = data.frame(class = temp[train,]$link_open,score = rf.link$predicted)
roc_rf = roc.curve(scores.class0=rf_scores[rf_scores$class==1,]$score,
          scores.class1=rf_scores[rf_scores$class==0,]$score,
          curve=T)
plot(roc_rf,col = 'dark red',xlab = 'Specificity' )


#####################################################logit


glm_link = glm(link_open~. , 
           data = temp[train,],
           family = binomial)

glm.probs = predict(glm_link,temp[-train,], type="response")
glm.pred=rep(0, nrow(temp[-train,]))
glm.pred[glm.probs >= cutoff] = 1

table(glm.pred,temp[-train,]$link_open)
mean(glm.pred==temp[-train,]$link_open)

test.err = mean((glm.probs-temp[-train,]$link_open)^2)
#0.02122791

glm_scores = data.frame(class = temp[train,]$link_open,score = glm_link$fitted.values)
roc_glm = roc.curve(scores.class0=rf_scores[glm_scores$class==1,]$score,
                   scores.class1=rf_scores[glm_scores$class==0,]$score,
                   curve=T)
plot(roc_glm,col = 'dark blue', ,xlab = 'Specificity',main = 'ROC curve',
     add = T)


#####################################################boosted trees


boosted.link = gbm(link_open~.,data = temp[train,], n.trees  = 5000, distribution = 'gaussian',
                   interaction.depth = 4, shrinkage = 0.1, verbose = T)
summary(boosted.link)

plot(boosted.link,i = 'hour',ylab = 'effect',main = 'boosted trees partial dependence plots')
plot(boosted.link,i = 'user_past_purchases',ylab = 'effect',main = 'boosted trees partial dependence plots')
plot(boosted.link,i = 'weekday',ylab = 'effect',main = 'boosted trees partial dependence plots')

conv_pred_boosted = predict(boosted.link,temp[-train,],n.trees = 5000)
boosted_pred= rep(0,nrow(temp[-train,]))
boosted_pred[conv_pred_boosted>0.5]=1
table(boosted_pred,temp[-train,]$link_open)
mean(boosted_pred==temp[-train,]$link_open)

boost_scores = data.frame(class = temp[train,]$link_open,score = boosted.link$fit)
roc_boost = roc.curve(scores.class0=boost_scores[boost_scores$class==1,]$score,
                      scores.class1=boost_scores[boost_scores$class==0,]$score,
                      curve=T)
plot(roc_boost,col = 'dark green', ,xlab = 'Specificity',main = 'ROC curve',
     add = T)


#####################################################compare


legend("topleft",c("random forest", "logit",'boosted trees'),
       fill=c("dark red", "dark blue",'dark green'))


##################################################### prediction comparison


n.trees = seq(from=100 ,to=5000, by=100) #no of trees-a vector of 100 values 

#Generating a Prediction matrix for each Tree
predmatrix<-predict(boosted.link,temp[-train,],n.trees = n.trees)
dim(predmatrix) #dimentions of the Prediction Matrix

#Calculating The Mean squared Test Error
test.error<-with(temp[-train,],apply( (predmatrix-link_open)^2,2,mean))
head(test.error) #contains the Mean squared test error for each of the 100 trees averaged

#Plotting the test error vs number of trees

plot(n.trees , test.error , pch=10,col="blue",xlab="Number of Trees",ylab="Test Error", 
     main = "Perfomance of Boosting on Test Set" ,ylim=c(0.021, 0.024))

#adding the RandomForests Minimum Error line trained on same data and similar parameters
abline(h = test.err.rf,col="red") #test.err is the test error of a Random forest fitted on same data

abline(h = test.err,col="green")
legend("topleft",c("Test error Line for logit","Test error Line for random forest"),
       fill=c("green", "red"),lty=1,lwd=1)

##################################################### optimization test


hour = link_hour$hour[link_hour$avg_ctr>0.02]
weekday = link_wk$weekday[link_wk$avg_ctr>0.02]
country =link_country$user_country[link_country$avg_ctr>0.02]
purchase = unique(df$user_past_purchases)
df_test = merge(hour,weekday,by = NULL)
df_test = merge(df_test,country,by = NULL)
df_test = merge(df_test,purchase,by = NULL)
names(df_test)=c('hour','weekday','user_country','user_past_purchases')
df_test$email_version = 'personalized'
df_test$email_text = 'short_email'
df_test$link_open = 0
df_test$hour = as.factor(df_test$hour)
glm.test = predict(glm_link,df_test,type = 'response')
mean(glm.test)#~19%


## Conclusion:
#After tuning the models and comparison of their prediction performance, 
#I decided to use logit function to predict and optimize. 
#To be more specific, I generated barplots and dependence plots to see which
#factors are the main reason driving today's result. I found out that people
#with higher past purchases and from US and UK, are more willingly to open the
#link attached in the email if they received the emails on selected hours and weekdays.
#In sum, the click
#through rate can be improved, per simulation, by about 17%. In order to test that
#the company should conduct a relatively long-term experiment, more specifically
#an A/B test. Randomly assign the user to two groups, treatment group(recieving
#modified email on selected hours and weekdays) and control group(recieving same
#type of email on the same hours and weekdays as the company is currently doing now.)