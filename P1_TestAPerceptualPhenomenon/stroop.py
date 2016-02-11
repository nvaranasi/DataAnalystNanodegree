# -*- coding: utf-8 -*-
"""
Created on Fri Jan 1 2016

@author: NVarana
"""

#File to test the Stroop Effect

import pandas as pd
import cmath

path = r'C:/Users/nvarana/Desktop/Training/DataAnalysisUdacity/data/stroop.csv'

d = pd.read_csv(path)

d

#Measures of central tendancy
meanCongruent = d['congruent'].mean()
meanIncongruent = d['incongruent'].mean()

#Measures of variability
varCongruent = d['congruent'].var()
varIncongruent = d['incongruent'].var()
stdCongruent = d['congruent'].std()
stdIncongruent = d['incongruent'].std()

#Plot distribution of sample data
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

#This command displays figures in their own window instead of the IPython console
%matplotlib qt
#Histogram of time taken to name list of congruent words and 
#time taken to name the list of incongruent words
s = d['congruent']
n, bins, patches = plt.hist(s, 10, normed=1, facecolor='blue', alpha=0.75)

#Add a best fit line
a = mlab.normpdf(bins, meanCongruent, stdCongruent)
l = plt.plot(bins, a, 'g--', linewidth=2)

plt.xlabel('Seconds per List of Words')
plt.ylabel('Probability')
plt.title('Histogram of Time Taken to Name Congruent Words')
plt.xlim([5,40])
plt.grid(True)
plt.show()

#Incongruent List of words
s = d['incongruent']
n, bins, patches = plt.hist(s, 10, normed=1, facecolor='red', alpha=0.75)

#Add a best fit line
b = mlab.normpdf(bins, meanIncongruent, stdIncongruent)
l = plt.plot(bins, b, 'g--', linewidth=2)

plt.xlabel('Seconds per List of Words')
plt.ylabel('Probability')
plt.title('Histogram of Time Taken to Name Incongruent Words')
plt.xlim([5, 40])
plt.grid(True)
plt.show()

#Test the hypothesis that the time taken to name the congruent list of words is less than the 
#time taken to name the list of incongruent words. 

#Find the difference in timing between naming congruent and incongruent words list
d['diff'] = d['congruent'] - d['incongruent']
d['meandiff'] = d['diff'].mean()
d['dev'] = (d['diff'] - d['meandiff'])
d['sqdev'] = d['dev']*d['dev']
dlen = len(d.index)
std_diff = cmath.sqrt(sum(d['sqdev']) / (dlen-1))
mean_diff = d['diff'].mean()

#Compute the dependent t-statistic for the paired samples
t = mean_diff / (std_diff / cmath.sqrt(dlen))

#For an alpha level of 0.05 and sample size of 24, 
#the t-critical value for a one-tailed t-test 
t_critical = -1.711

#Since our t-statistic is less than t critical value, we can reject the null that the time taken to name 
#the coherent words list is greater than or equal to the time taken to name the incoherent words list

#Compute confidence interval 
ci_lower = mean_diff - t_critical * (std_diff/cmath.sqrt(dlen))
ci_upper = mean_diff + t_critical * (std_diff/cmath.sqrt(dlen))

























 
