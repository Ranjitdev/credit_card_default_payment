## Overview of Credit Card Default Payment
**Financial threats are displaying a trend about the credit risk of commercial banks as the incredible improvement in the financial industry has arisen. In this way, one of the biggest threats faces by commercial banks is the risk prediction of credit clients. The goal is to predict the probability of credit default based on credit card owner's characteristics and payment history.**

**A machine learning credit card default project is a data science project that aims to predict credit card default risk using machine learning techniques. Credit card default occurs when a credit cardholder fails to make the required minimum payment on their credit card balance for a specified period, usually 30 days or more.**

**The objective of the project is to build a predictive model that can assess the probability of a credit card user defaulting on their payments. This model can be valuable to banks, financial institutions, and credit card companies as it helps them identify customers who are more likely to default and take appropriate actions to mitigate the risks.**

## About Dataset
- Link : [Here](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- Dataset Information
  - This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.
    - **Pay : Repayment status**
    - **Bill : Amount of bill statement**
    - **Paid : Amount of previous payment**
    - **-2 = Balance paid in full and no transactions this period (we may refer to this credit card account as having been 'inactive' this period**
    - **-1 = Balance paid in full, but account has a positive balance at end of period due to recent transactions for which payment has not yet come due**
    - **0 = Customer paid the minimum due amount, but not the entire balance. I.e., the customer paid enough for their account to remain in good standing, but did revolve a balance**
    - **1 = payment delay for one month**
    - **2 = payment delay for two months**
    . 
    . 
    .
    - **8 = payment delay for eight months**
    - **9 = payment delay for nine months and above.**

## Common Setup and Commands
- Create virtual enviroment:
  - > conda create -p venv python=3.9 -y
- Activate the virtual enviroment
  - > conda activate venv/
- Install required packages
  - > pip install -r requirements.txt
- Install setup
  - > python setup.py install