# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 09:25:19 2017

@author: YKrueng
"""

# This code is used to performed a repeated measured t-test on Stroop Task dataset
# on a Project: Test a Perceptual Phenomenon as part of Data Analysis Nanodegree Program at Udacity
# Though other python modules exist to do the task more efficiently, this code is intended 
# to demonstrate the step-by-step procedure and calculation of t-test.  
# Methods to plot a scatter plot and a histogram are also provided to do descriptive analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import t
import scipy.stats as st
import math
import seaborn as sns

# load dataset and add colum for the differences
# returns dataset in DataFrame
def loadDataSet():
    data = pd.read_csv('stroopdata.csv')
    data.index = range(1,25)
    data['Differences'] = data['Congruent']-data['Incongruent']
    return data
    
# perform t-test on sample differences with significant level of alpha
# tail=2 for two-tailed test and tail=1 for one-tailed test
# direction = 1 for positive and direction = -1 for negative (only apply in one-tailed test)
# returns cohen's d, r_squared, and t-test result
def ttest(xd, alpha,tail=1, direction = 1):
    index = ["t-test"]
    header = ["mean", "std", "df", "t", "t_critical", "p", "type"]
    
    # configure test condition
    if tail==2:
        alpha = alpha/2
        m=1
        label = "two-tailed"
    elif tail==1 and direction==-1:
        m=-1
        label = "one-tailed, neg"
    elif tail==1 and direction==1:
        m=1
        label = "one-tailed, pos"
        
    # sample size
    n = len(xd)
    
    # degree of freedom
    df = n-1
    
    # mean of sample differences
    mean = np.mean(xd)
    
    # standard deviation of sample differences
    std = np.std(xd, ddof=1)
    
    # t-value
    t_v = mean/(std/math.sqrt(n))
    
    # t-critical
    t_c = m*t.ppf(1 - alpha, df)
    
    # p-value
    p = tail*t.sf(abs(t_v), df)
    
    result = pd.DataFrame(np.matrix([mean,std, df, t_v, t_c, p, label]),
                          index,
                          header)
    
    # cohen's d
    d = mean/std
    
    # r squared
    r_squared = t_v*t_v/(t_v*t_v+df)
    
    return d, r_squared, result    

# determine the confidence interval of sample differences
# returns confidence interval
def ci(xd, conf=0.99):
    mean = np.mean(xd)
    me = np.std(xd, ddof=1)/math.sqrt(len(xd))
    t_c = t.ppf(conf+((1-conf)/2), len(xd)-1)
    ci = [mean-t_c*me, mean+t_c*me]
    return ci
    
# draw a histrogram of sample differences    
def hist(xd):
    plt.hist(xd, bins=range(-25, 0+5, 5))

    plt.xticks(np.arange(-25,0+5,5))
    plt.xlim(-25,0)
    
    plt.title("Stroop Task Finish Time Differences")
    plt.xlabel("Finish Time Difference (seconds)")
    plt.ylabel("Frequency")
    plt.show()
    
    print "Skewness = " + str(st.skew(xd))
    print "Kurtosis = " + str(st.kurtosis(xd))
    


# draw a scatter plot
def scatter(x1,x2):
    plt.scatter(x1,x2)
    plt.title("Time to Complete Stroop Task in Seconds")
    plt.xlabel("Congruent Words")
    plt.ylabel("Incongruent Words")
    plt.show()
    
    # calculate Pearson's r
    r, p = st.pearsonr(x1,x2)
    print "Pearson's r = " + str(r)

# main method
if __name__ == "__main__":
    # load data
    data = loadDataSet()
    
    # display data statistics
    print data.describe()
    print "" 
    
    # draw plots for descriptive analysis
    scatter(data.Congruent, data.Incongruent)
    hist(data.Differences)
    
    # conduct a negative one-tailed repeated measures t-test with alpha = .001 
    d,r_squared,result = ttest(data.Differences, .001, tail=1, direction=-1)
    
    print result
    print ""
    print "Cohen's d = " + str(d)
    print "r_squared = " + str(r_squared)
    print ""
    
    # determine confidence interval of mean sample differences with confidence
    # level of 99.9%
    ci = ci(data.Differences, .999)
    print "99% Confidence Interval of mean differences = " + str(ci) 