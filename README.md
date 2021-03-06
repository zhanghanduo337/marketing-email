# marketing-email
Optimizing marketing campaigns

## Challenge Description

The marketing team of an e-commerce site has launched an email campaign. This site has email addresses from all the users who created an account in the past.
**They have chosen a random sample of users and emailed them.** The email let the user know about a new feature implemented on the site. From the marketing team perspective, **a success is if the user clicks on the link inside of the email**. This link takes the user to the company site.
You are in charge of figuring out how the email campaign performed and were asked the following questions:

1. **What percentage of users** opened the email and what percentage clicked on the link within the email?
2. The VP of marketing thinks that it is stupid to send emails to a random subset and in a random way. Based on all the information you have about the emails that were sent, can you **build a model to optimize in future email campaigns to maximize the probability of users clicking on the link inside the email?**
3. **By how much do you think your model would improve click through rate** ( defined as # of users who click on the link / total users who received the email). How would you test that?
4. Did you find any **interesting pattern** on how the email campaign performed for different segments of users? Explain.

Note:
comments are embedded in the code file

### Some highlights from the project


![click through rate pattern chart](CTR.png)
![ROC curve of three models](ROC_curve.png)
![prediction performance comparison](performance_comparison.png)
![boost_dependence](boost_dependence.png)


## Conclusion:
After tuning the models and comparison of their prediction performance, 
I decided to use logit function to predict and optimize. 
To be more specific, I generated barplots and dependence plots to see which
factors are the main reason driving today's result. I found out that people
with higher past purchases and from US and UK, are more willingly to open the
link attached in the email if they received the emails on selected hours and weekdays.
In sum, the click
through rate can be improved, per simulation, by about 17%. In order to test that
the company should conduct a relatively long-term experiment, more specifically
an A/B test. Randomly assign the user to two groups, treatment group(recieving
modified email on selected hours and weekdays) and control group(recieving same
type of email on the same hours and weekdays as the company is currently doing now.)
