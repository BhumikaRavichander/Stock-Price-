#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Implementing an artificial recurrent neural network called Long Short Term Memory (LSTM) to predict the closing stock price
#    of a corporation (Apple Inc.) using the stock prices of the last 60 days.


# In[2]:


#import libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential 
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[7]:


#Getting the stock quote and displaying it
df=web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')
df


# In[8]:


#visualize the closing price
plt.figure(figsize=(16,8))
plt.title('Close Price Prediction')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD($)', fontsize=18)
plt.show()


# In[11]:


#Create a new dataframe with only the 'Close' column
data=df.filter(['Close'])

#Convert the dataset into a numpy array
dataset=data.values

#Get the number of rows to train the model on
training_data_length=math.ceil(len(dataset) * .8)
training_data_length


# In[13]:


#Scale the data
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
scaled_data


# In[18]:


#create the scaled training dataset
train_data=scaled_data[0:training_data_length,:]

#split the data into x_train and y_train datasets
x_train =[]
y_train=[]

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    
    if i<=60:
        print(x_train)
        print(y_train)
        print()


# In[19]:


#convert the x_train and y_train into numpy arrays
x_train,y_train=np.array(x_train), np.array(y_train)


# In[23]:


#Reshape the data
x_train=np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))
x_train.shape


# In[25]:


#Build the LSTM model
model=Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[26]:


#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[27]:


#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[35]:


#Create the test dataset

#Create a new array containing scaled values from 1543 to 2003
test_data=scaled_data[training_data_length - 60:, :]

#create datasets x_test and y_test
x_test = []
y_test = dataset[training_data_length:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    


# In[36]:


#Convert data into numpy array
x_test=np.array(x_test)


# In[37]:


#Reshape the array
x_test=np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))
x_test.shape


# In[39]:


#Get the model's predicted price values
predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)


# In[40]:


#Get the root mean squared error
rmse=np.sqrt(np.mean(predictions - y_test) **2 )
rmse


# In[44]:


#Plot the data
train = data[:training_data_length]
valid = data[training_data_length :]
valid['Predictions']=predictions

#Visualize
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train', 'Val','Predictions'], loc="lower right")
plt.show()


# In[45]:


#Show the valid and predicted prices
valid


# In[50]:


#Get the quote
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')

#create a new dataframe
new_df = apple_quote.filter(['Close'])

#Get the last 60 day closing price values and convert the dataframe to an array
last_60_days=new_df[-60:].values

#Scale the data to values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)

#create an empty list
X_test=[]

#append the past 60 days
X_test.append(last_60_days_scaled)

#Convert the X_test data set into numpy array
X_test=np.array(X_test)

#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

#Get the predicted scaled price
pred_price=model.predict(X_test)

#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)


# In[51]:


#Get the quote
apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2019-12-18', end='2019-12-18')
print(apple_quote2['Close'])


# In[ ]:




