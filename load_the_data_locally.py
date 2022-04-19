import pandas as pd
import os
from datetime import datetime
import numpy as np

def csv_reader_func(filename,symbol):
    
    pdf = None
    if os.path.exists(os.path.join(os.getcwd(),filename)):
        if os.path.getsize(os.path.join(os.getcwd(),filename)) != 0:
            pdf = pd.read_csv(os.path.join(os.getcwd(),filename), header = 'infer')   
            pdf['symbol'] = symbol
            
            if pdf.empty == True:
                return 0
            else:
                return pdf
        else:
            return 0
    else:
        return 0
    

def parse_datestring(date_string = None, str_pattern = None) :
    return datetime.strptime(date_string, str_pattern).date()


# formate numbers from string format e.g. 1,200.00 to the type of choice 
def format_to(type, x):
    if x.__class__ == str:
        if ',' in list(x):
            x = x.replace(',','')
        if x.replace('.','').isnumeric() == False:
            return None	
    return type(x)


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

    

def load_data():
    """
       Loads the data in pandas dataframes and applies some preprocessing to them.
       Input:
       Output:
              symbols : pandas dataframe with the symbols of the stocks of the portfolio
              stocks : pandas dataframe with the stock data
              factors : pandas dataframe with the factor data
       
    """
    symbols_filename = "Spark/projects/risk_management/data/symbols.txt"
    stocks_dir = "Spark/projects/risk_management/data/stocks"
    factors_dir = "Spark/projects/risk_management/data/factors"
    """  read the 'symbols'   """
    symbols = pd.read_table(symbols_filename, names = ['value'])
    
    """  read the 'stocks'   """
    stocks = pd.DataFrame()
    for s in symbols['value']:
        pdf = csv_reader_func(f"{stocks_dir}/{s}.csv",s)
        if type(pdf)!=int:
            stocks =  stocks.append(pdf)
    """  read the 'factors'   """
    factor_columns = ['Date','Open','High','Low','Close','Adj Close','Volume','symbol']
    snp = pd.read_table(factors_dir+"/GSPC.txt",header = 'infer',sep = "\t",converters = {'Date':lambda x: parse_datestring(date_string = x,str_pattern = '%b %d, %Y' ),'Open':lambda x : format_to(float,x),'High':lambda x : format_to(float,x),'Low':lambda x : format_to(float,x),'Close*':lambda x : format_to(float,x),'Adj Close**':lambda x : format_to(float,x),'Volume':lambda x : format_to(np.long,x)})
    snp['symbol'] = "gspc"
    snp.columns = factor_columns 
     
    nasdaq = pd.read_csv(factors_dir+"/IXIC.csv",header = 'infer',converters = {'Date':lambda x: parse_datestring(date_string = x,str_pattern = '%Y-%m-%d' ),'Open':lambda x : format_to(float,x),'High':lambda x : format_to(float,x),'Low':lambda x : format_to(float,x),'Close':lambda x : format_to(float,x),'Adj Close':lambda x : format_to(float,x),'Volume':lambda x : format_to(np.long,x)})
    nasdaq['symbol'] = "ixic"
    nasdaq.columns = factor_columns 
    
    bonds5y = pd.read_csv(factors_dir+"/FVX.csv",header = 'infer',converters = {'Date':lambda x: parse_datestring(date_string = x,str_pattern = '%Y-%m-%d' ),'Open':lambda x : format_to(float,x),'High':lambda x : format_to(float,x),'Low':lambda x : format_to(float,x),'Close':lambda x : format_to(float,x),'Adj Close':lambda x : format_to(float,x),'Volume':lambda x : format_to(np.long,x)})
    bonds5y['symbol']="fvx"
    bonds5y.columns = factor_columns 
    # filling null policy fills null value for the factor the most recent
    bonds5y = bonds5y.apply(fillnulls_with_recent, axis = 0)
    
    bonds30y = pd.read_csv(factors_dir+"/TYX.csv",header = 'infer',converters = {'Date':lambda x: parse_datestring(date_string = x,str_pattern = '%Y-%m-%d' ),'Open':lambda x : format_to(float,x),'High':lambda x : format_to(float,x),'Low':lambda x : format_to(float,x),'Close':lambda x : format_to(float,x),'Adj Close':lambda x : format_to(float,x),'Volume':lambda x : format_to(np.long,x)})
    bonds30y['symbol']="tyx"
    bonds30y.columns = factor_columns
    bonds30y = bonds30y.apply(fillnulls_with_recent, axis = 0)	     

    factors = snp
    factors = factors.append([nasdaq, bonds5y,bonds30y], ignore_index = True)             

    return symbols, stocks, factors