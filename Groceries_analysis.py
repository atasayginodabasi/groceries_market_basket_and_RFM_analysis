import os
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import operator as op
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Haftalık olarak unique userların sayısını veriyor (lazım olabilir)
'''''''''
x = data.resample('w', on='Created_on').Website_Connection_Log.nunique()
'''''''''

# Reading the data
print(os.listdir(r'C:/Users/ata-d/'))
data = pd.read_csv(r'C:/Users/ata-d/OneDrive/Masaüstü/ML/Datasets/Groceries_dataset.csv')

data.columns = ['memberID', 'Date', 'itemName']

data.info()  # Date data-type is not in correct from, it must be transformed into datetime type

# ----------------------------------------------------------------------------------------------------------------------

# Converting Date column into correct datatype which is datetime
data.Date = pd.to_datetime(data.Date)
data.memberID = data['memberID'].astype('str')

# ----------------------------------------------------------------------------------------------------------------------

# Number of Sales Weekly
'''''''''
Sales_weekly = data.resample('w', on='Date').size()
fig = px.line(data, x=Sales_weekly.index, y=Sales_weekly,
              labels={'y': 'Number of Sales'})
fig.update_layout(title_text='Number of Sales Weekly',
                  title_x=0.5, title_font=dict(size=20)) 
fig.show()
'''''''''

# Number of Customers Weekly
'''''''''
Unique_customer_weekly = data.resample('w', on='Date').memberID.nunique()
fig = px.line(Unique_customer_weekly, x=Unique_customer_weekly.index, y=Unique_customer_weekly,
              labels={'y': 'Number of Customers'})
fig.update_layout(title_text='Number of Customers Weekly',
                  title_x=0.5, title_font=dict(size=20))
fig.show()
'''''''''

# Sales per Customer Weekly
'''''''''
Sales_per_Customer = Sales_weekly / Unique_customer_weekly
fig = px.line(Sales_per_Customer, x=Sales_per_Customer.index, y=Sales_per_Customer,
              labels={'y': 'Sales per Customer Ratio'})
fig.update_layout(title_text='Sales per Customer Weekly',
                  title_x=0.5, title_font=dict(size=20))
fig.update_yaxes(rangemode="tozero")
fig.show()
'''''''''

# Frequency of the Items Sold
'''''''''
Frequency_of_items = data.groupby(pd.Grouper(key='itemName')).size().reset_index(name='count')
fig = px.treemap(Frequency_of_items, path=['itemName'], values='count')
fig.update_layout(title_text='Frequency of the Items Sold',
                  title_x=0.5, title_font=dict(size=20)
                  )
fig.update_traces(textinfo="label+value")
fig.show()
'''''''''

# Top Customers regarding Number of Items bought
'''''''''
user_item = data.groupby(pd.Grouper(key='memberID')).size().reset_index(name='count') \
    .sort_values(by='count', ascending=False)
fig = px.bar(user_item.head(25), x='memberID', y='count',
             labels={'y': 'Number of Sales',
                     'count': 'Number of Items Bought'},
             color='count')
fig.update_layout(title_text='Top 20 Customers regarding Number of Items Bought',
                  title_x=0.5, title_font=dict(size=20))
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))

fig.show()
'''''''''

# Number of Sales per Discrete Week Days
'''''''''
day = data.groupby(data['Date'].dt.strftime('%A'))['itemName'].count()
fig = px.bar(day, x=day.index, y=day, color=day,
             labels={'y': 'Number of Sales',
                     'Date': 'Week Days'})
fig.update_layout(title_text='Number of Sales per Discrete Week Days',
                  title_x=0.5, title_font=dict(size=20))
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
fig.show()
'''''''''

# Number of Sales per Discrete Months
'''''''''
month = data.groupby(data['Date'].dt.strftime('%m'))['itemName'].count()
fig = px.bar(month, x=month.index, y=month, color=month,
             labels={'y': 'Number of Sales',
                     'Date': 'Months'})
fig.update_layout(title_text='Number of Sales per Discrete Months',
                  title_x=0.5, title_font=dict(size=20))
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
fig.show()
'''''''''

# Number of Sales per Discrete Month Days
'''''''''
month_day = data.groupby(data['Date'].dt.strftime('%d'))['itemName'].count()
fig = px.bar(month_day, x=month_day.index, y=month_day, color=month_day,
             labels={'y': 'Number of Sales',
                     'Date': 'Month Days'})
fig.update_layout(title_text='Number of Sales per Discrete Month Days',
                  title_x=0.5, title_font=dict(size=20))
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
fig.show()
'''''''''

# ----------------------------------------------------------------------------------------------------------------------

basket = data.groupby(['memberID', 'itemName'])['itemName'].count().unstack().fillna(0).reset_index()


def one_hot_encoder(k):
    if k <= 0:
        return 0
    if k >= 1:
        return 1


basket_final = basket.iloc[:, 1:basket.shape[1]].applymap(one_hot_encoder)

# ----------------------------------------------------------------------------------------------------------------------

frequent_itemsets = apriori(basket_final, min_support=0.025, use_colnames=True, max_len=3)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1).sort_values('lift', ascending=False)
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

# ----------------------------------------------------------------------------------------------------------------------

# RECENCY

# Finding last purchase date of each customer
Recency = data.groupby(by='memberID')['Date'].max().reset_index()
Recency.columns = ['memberID', 'LastDate']

# Finding last date for our dataset
last_date_dataset = Recency['LastDate'].max()

# Calculating Recency by subtracting (last transaction date of dataset) and (last purchase date of each customer)
Recency['Recency'] = Recency['LastDate'].apply(lambda x: (last_date_dataset - x).days)

# Recency of the Customers
'''''''''
fig = px.histogram(Recency, x='Recency', opacity=0.85, marginal='box')
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
fig.update_layout(title_text='Recency of the Customers',
                  title_x=0.5, title_font=dict(size=20))
fig.show()
'''''''''

# ----------------------------------------------------------------------------------------------------------------------

# FREQUENCY

# Frequency of the customer visits
Frequency = data.drop_duplicates(['Date', 'memberID']).groupby(by=['memberID'])['Date'].count().reset_index()
Frequency.columns = ['memberID', 'Visit_Frequency']

# Visit Frequency of the Customers
'''''''''
fig = px.histogram(Frequency, x='Visit_Frequency', opacity=0.85, marginal='box')
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
fig.update_layout(title_text='Visit Frequency of the Customers',
                  title_x=0.5, title_font=dict(size=20))
fig.show()
'''''''''

# ----------------------------------------------------------------------------------------------------------------------

# MONETARY

Monetary = data.groupby(by="memberID")['itemName'].count().reset_index()
Monetary.columns = ['memberID', 'Monetary']

# I assumed each item has equal price and price is 10
Monetary['Monetary'] = Monetary['Monetary'] * 10

# Monetary of the Customers
'''''''''
fig = px.histogram(Monetary, x='Monetary', opacity=0.85, marginal='box',
                   labels={'itemName': 'Monetary'})
fig.update_traces(marker=dict(line=dict(color='#000000', width=1)))
fig.update_layout(title_text='Monetary of the Customers',
                  title_x=0.5, title_font=dict(size=20))
fig.show()
'''''''''

# ----------------------------------------------------------------------------------------------------------------------

RFM = pd.concat([Recency['memberID'], Recency['Recency'], Frequency['Visit_Frequency'], Monetary['Monetary']], axis=1)

# 5-5-5 score = the best customers
RFM['Recency_quartile'] = pd.qcut(RFM['Recency'], 5, [5, 4, 3, 2, 1])
RFM['Frequency_quartile'] = pd.qcut(RFM['Visit_Frequency'], 5, [1, 2, 3, 4, 5])

RFM['RF_Score'] = RFM['Recency_quartile'].astype(str) + \
                   RFM['Frequency_quartile'].astype(str)

# ----------------------------------------------------------------------------------------------------------------------

segt_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at risk',
    r'[1-2]5': 'can\'t loose',
    r'3[1-2]': 'about to sleep',
    r'33': 'need attention',
    r'[3-4][4-5]': 'loyal customers',
    r'41': 'promising',
    r'51': 'new customers',
    r'[4-5][2-3]': 'potential loyalists',
    r'5[4-5]': 'champions'
}

RFM['RF_Segment'] = RFM['RF_Score'].replace(segt_map, regex=True)

# ----------------------------------------------------------------------------------------------------------------------

# Distribution of the RFM Segments
'''''''''
x = RFM.RF_Segment.value_counts()
fig = px.treemap(x, path=[x.index], values=x)
fig.update_layout(title_text='Distribution of the RFM Segments', title_x=0.5,
                  title_font=dict(size=20))
fig.update_traces(textinfo="label+value+percent root")
fig.show()
'''''''''

# Relationship between Visit_Frequency and Recency
'''''''''
fig = px.scatter(RFM, x="Visit_Frequency", y="Recency", color='RF_Segment',
                 labels={"math score": "Math Score",
                         "writing score": "Writing Score"})
fig.update_layout(title_text='Relationship between Visit_Frequency and Recency',
                  title_x=0.5, title_font=dict(size=20))
fig.show()
'''''''''
