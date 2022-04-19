import numpy as np
import pandas as pd

# fill null values with the most recent and if the first value is null replace
# the null values with the mean value of the series data
def fillnulls_with_recent(ser):  
    serindex = ser.index
    new_index = np.arange(len(ser)).tolist()
    ser.index = new_index
    ser = ser.bfill()
    ser = ser.fillna(ser.mean())
    ser.index = serindex
    
    return ser

    
    
# return days in seconds
def days(days):
    return  days * 24 * 3600
      
      
def average(l):
    if len(l)!= 0:
        return sum(l)/len(l)
    else:
        return 0      