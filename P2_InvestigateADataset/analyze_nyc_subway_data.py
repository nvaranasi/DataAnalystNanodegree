# -*- coding: utf-8 -*-
"""
Created on Jan 29 2016

@author: NVarana
"""

#Code to Analyze NYC Subway Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import cmath
import scipy.stats as ss
import statsmodels.api as sm
from patsy import dmatrices

path = r'C:/Users/nvarana/Desktop/Training/DataAnalysisUdacity/AnalyzeNYCData/'

##Section 1: Statistical Test
subway_df = pd.read_csv(path+'turnstile_data_master_with_weather.csv')
subway_df.head()
subway_df.describe()

#Split the dataset into the with_rain and without_rain dataset
with_rain = subway_df[subway_df['rain']==1] 
without_rain = subway_df[subway_df['rain']==0] 

#Plot histogram of ridership when its raining versus when its not
%matplotlib qt
plt.figure
with_rain['ENTRIESn_hourly'].hist(label="Rain", bins=100)
#plt.legend(loc="upper right")
plt.xlabel('Hourly Turnstile Entries')
plt.xlim([0,5000])
plt.ylabel('Frequency')
plt.title('Histogram of Hourly Ridership - Rain')
plt.show()

%matplotlib qt
without_rain['ENTRIESn_hourly'].hist(label="No Rain", bins=100)
plt.xlabel('Hourly Turnstile Entries')
plt.xlim([0,5000])
plt.ylabel('Frequency')
plt.title('Histogram of Hourly Ridership - No Rain')
plt.show()

#Mann-Whitney 
u,p = ss.mannwhitneyu(with_rain['ENTRIESn_hourly'], without_rain['ENTRIESn_hourly'])

#Mean ridership with rain and without rain
avg_ridership_rain = np.mean(with_rain[['ENTRIESn_hourly']])
avg_ridership_without_rain = np.mean(without_rain[['ENTRIESn_hourly']])

#ridership_rain = with_rain.groupby(['DATEn'], as_index=False)['ENTRIESn_hourly'].sum()
#ridership_without_rain = without_rain.groupby(['DATEn'], as_index=False)['ENTRIESn_hourly'].sum()
#avg_riders_rain = np.mean(ridership_rain['ENTRIESn_hourly'])
#avg_riders_without_rain = np.mean(ridership_without_rain['ENTRIESn_hourly'])



##Section 2: Linear Regression

df = pd.read_csv(path+'turnstile_weather_v2.csv')

#Exploratory analyses of NYC subway data
#Ridership by day
ridership_by_day = df.groupby('day_week').mean()['ENTRIESn_hourly']

%matplotlib qt
plt.figure
ridership_by_day.plot(kind='bar', title='Ridership By Day Of Week')
plt.xlabel("Day Of Week")
plt.ylabel("Average Ridership Per Hour")
plt.show()

#Ridership by hour of hour
ridership_by_hour = df.groupby('hour').mean()['ENTRIESn_hourly']

%matplotlib qt
plt.figure
ridership_by_hour.plot(title='Ridership By Hour Of Day')
plt.xlabel('Hour Of Day')
plt.ylabel("Average Ridership")
plt.show()

#Split the dataset into the with_fog and without_fog dataset
with_fog = df[df['fog']==1] 
without_fog = df[df['fog']==0] 

##Not using fog as a feature since with_fog sample is only 1%, without_sample is 99% 

##Which UNITS have the most riders and how different are the riderships across these UNITS?
ridership_by_unit= df.groupby('UNIT')['ENTRIESn_hourly'].mean()
ridership_by_unit.describe()

##Conditions for the time and location 
ridership_by_cond = df.groupby('conds')['ENTRIESn_hourly'].mean()
%matplotlib qt
plt.figure
ridership_by_cond.plot(kind='bar', title='Ridership By Condition')
plt.xlabel("Condition")
plt.ylabel("Avg. Ridership Per Hour")
plt.show()

##Ridership by Rain? 
ss.pearsonr(df['ENTRIESn_hourly'], df['rain'])

##Ridership by Fog? 
ss.pearsonr(df['ENTRIESn_hourly'], df['fog'])

##Ridership by Temperature? 
ss.pearsonr(df['ENTRIESn_hourly'], df['tempi'])
ss.pearsonr(df['ENTRIESn_hourly'], df['meantempi'])

##Ridership by Wind Speed? 
ss.pearsonr(df['ENTRIESn_hourly'], df['wspdi'])
ss.pearsonr(df['ENTRIESn_hourly'], df['meanwspdi'])

##Ridership by Precipitation? 
ss.pearsonr(df['ENTRIESn_hourly'], df['precipi'])
ss.pearsonr(df['ENTRIESn_hourly'], df['meanprecipi'])

##Ridership by Barometric pressure? 
ss.pearsonr(df['ENTRIESn_hourly'], df['pressurei'])
ss.pearsonr(df['ENTRIESn_hourly'], df['meanpressurei'])

##Ridership by latitude and longitude
ss.pearsonr(df['ENTRIESn_hourly'], df['latitude'])
ss.pearsonr(df['ENTRIESn_hourly'], df['longitude'])

pred = predictions(df)
#%matplotlib qt
plot_residuals(df, pred)       

def predictions(dataframe):
    '''
    The NYC turnstile data is stored in a pandas dataframe called weather_turnstile.
    Using the information stored in the dataframe, let's predict the ridership of
    the NYC subway using linear regression with ordinary least squares.
    
    You can download the complete turnstile weather dataframe here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv    
    
    Your prediction should have a R^2 value of 0.40 or better.
    You need to experiment using various input features contained in the dataframe. 
    We recommend that you don't use the EXITSn_hourly feature as an input to the 
    linear model because we cannot use it as a predictor: we cannot use exits 
    counts as a way to predict entry counts. 
    
    Note: Due to the memory and CPU limitation of our Amazon EC2 instance, we will
    give you a random subet (~10%) of the data contained in 
    turnstile_data_master_with_weather.csv. You are encouraged to experiment with 
    this exercise on your own computer, locally. If you do, you may want to complete Exercise
    8 using gradient descent, or limit your number of features to 10 or so, since ordinary
    least squares can be very slow for a large number of features.
    
    If you receive a "server has encountered an error" message, that means you are 
    hitting the 30-second limit that's placed on running your program. Try using a
    smaller number of features.
    '''
    ################################ MODIFY THIS SECTION #####################################
    # Select features. You should modify this section to try different features!             #
    # We've selected rain, precipi, Hour, meantempi, and UNIT (as a dummy) to start you off. #
    # See this page for more info about dummy variables:                                     #
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html          #
    ##########################################################################################
    features = dataframe[['weekday', 'rain', 'meantempi', 'meanwspdi', 'meanprecipi', 'meanpressurei', 'tempi', 'wspdi', 'precipi', 'pressurei']]
    
    #y, X = dmatrices('ENTRIESn_hourly ~ weekday+ rain+ meantempi+ meanwspdi+ meanprecipi+ meanpressurei+ tempi+ wspdi+ precipi + pressurei+ UNIT+ conds', data=dataframe, return_type='dataframe')
    dummy_units = pd.get_dummies(dataframe['UNIT'], prefix='u')
    dummy_conds = pd.get_dummies(dataframe['conds'], prefix='conds')
    #dummy_conds = dummy_conds.drop('conds_Clear', 1)
    dummy_hr = pd.get_dummies(dataframe['hour'], prefix='hr')
    features = features.join(dummy_units)
    features = features.join(dummy_conds)
    features = features.join(dummy_hr)
    
    # Values
    values = dataframe['ENTRIESn_hourly']

    # Perform linear regression
    intercept, params = linear_regression(features, values)
    print params
    
    predictions = intercept + np.dot(features, params)
    predictions
    print compute_r_squared(values, predictions)    
    plot_residuals(df, predictions)       
    return predictions
 
 
def linear_regression(features, values):
    features = sm.add_constant(features)
    model = sm.OLS(values, features)
    results = model.fit()
    intercept = results.params[0]
    params = results.params[1:]
    return intercept, params

##Function to compute R-squared
def compute_r_squared(data, predictions):
    # Write a function that, given two input numpy arrays, 'data', and 'predictions,'
    # returns the coefficient of determination, R^2, for the model that produced 
    # predictions.
    # 
    # Numpy has a couple of functions -- np.mean() and np.sum() --
    # that you might find useful, but you don't have to use them.

    # YOUR CODE GOES HERE
    SST = ((data - np.mean(data))**2).sum()
    SSReg = ((predictions-data)**2).sum()
    r_squared =1 - SSReg / SST

    return r_squared

def plot_residuals(dataframe, predictions):
    '''
    Using the same methods that we used to plot a histogram of entries
    per hour for our data, why don't you make a histogram of the residuals
    (that is, the difference between the original hourly entry data and the predicted values).
    Try different binwidths for your histogram.

    Based on this residual histogram, do you have any insight into how our model
    performed?  Reading a bit on this webpage might be useful:

    http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm
    '''
    
    plt.figure()
    (dataframe['ENTRIESn_hourly'] - predictions).describe()
    (dataframe['ENTRIESn_hourly'] - predictions).hist()
    return plt
    

