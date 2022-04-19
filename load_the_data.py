
import pandas as pd

import os
from datetime import datetime, timedelta

from pyspark.sql.types import StringType, FloatType, DateType, IntegerType, StructType, TimestampType
import pyspark.sql.functions as F 

symbols_filename = os.getcwd()+"/risk_assessment/data/symbols.txt"       #   provide the location of the symbols.txt
stocks_dir = os.getcwd()+"/risk_assessment/data/stocks"                #   ensure that the project is in your current working directory or specify accordingly the location of 
                                                                       #   the folder where the 'stocks' data exists
factors_dir = os.getcwd()+"/risk_assessment/data/factors"		       #   ensure that the project is in your current working directory or specify accordingly the location of 
                                                                       #   the folder where the 'factor' data exists
def csv_reader_func(filename,symbol):
    pdf = None
    if os.path.exists(os.path.join(os.getcwd(),filename)):
        if os.path.getsize(os.path.join(os.getcwd(),filename)) != 0:
            pdf = pd.read_csv(os.path.join(os.getcwd(),filename), header = 'infer', sep = ',')   
            pdf['symbol'] = symbol
            
            if pdf.empty == True:
                return 0
            else:
                #pdf_pc = pickle.dumps(pdf)
                return pdf.values
        else:
            return 0
    else:
        return 0
        

def parse_datestring(date_string = None, str_pattern = None) :
    return datetime.strptime(date_string, str_pattern).date().strftime('%Y-%m-%d')


# formate numbers from string format e.g. 1,200.00 to the type of choice 
def format_to(type, x):
    if x == None:
        return None
    else:
        if x.__class__ == str:
            if ',' in list(x):
                x = x.replace(',','')
            if x.replace('.','').isnumeric() == False:
                return None	
        return type(x)

def load_the_data(spark):   
    """
       Loads the data and performs some initial preprocessing (converting to the right schema of types, dropping null values e.t.c.)
       Finds the data folder itself within the folder of the project.
       
       Input : 
       Output : 
               stocks : a spark dataframe with the stocks of the portfolio
               factors_dict : a dict ({factor_name : factor_data,}) with the factor data
       
    """
    sc = spark.sparkContext
    print('Loading the data....')																	
    """  read the symbols   """
    symbols_rdd = sc.parallelize([l.strip() for l in open(symbols_filename).readlines()])
    symbols = spark.createDataFrame(symbols_rdd, StringType(),['value'])
    symbols = symbols.repartition(100)
    symbols.cache()
    """ read the stocks  """
    columns = ['Date','Open','High','Low','Close','Volume','symbol']
    # Make an rdd containing pairs of the stock's symbol and the dataset for that stock. The rdd will contain
    # only stocks with non empty datasets of stock values.
    temp = symbols.rdd.map(lambda sym : csv_reader_func(f"{stocks_dir}/{sym.value}.csv",sym.value)).filter(lambda r : type(r) != int)
    temp = temp.repartition(100)
    temp = temp.flatMap(lambda ar : [list(ar[i,:]) for i in range(ar.shape[0])])
    """ Carefull!! Defining the df through the command, stocks = temp.toDF(columns), without specifying the schema, it can ruin a lot of things.
       Some entries are not justified by the schema inferred and are set to null. So, you loose information in this way, which may be difficult to 
    identify when working with a vast amount of data. So, we 'd rather use the 'createDataFrame' instruction and define the schema as generic
    as possible.
    """  
    schema = StructType().add('Date', StringType(), True).add('Open', StringType(), True).add('High', StringType(), True).add('Low', StringType(), True).add('Close', StringType(), True).add('Volume', StringType(), True).add('symbol', StringType(), True)
    stocks = spark.createDataFrame(temp, schema, columns)
    # type conversions for some fields
    stocks = stocks.withColumn('Date',F.udf(lambda datestr : parse_datestring(date_string = datestr, str_pattern = '%d-%b-%y'))(stocks.Date).cast('date'))
    stocks = stocks.withColumn('Open',F.udf(lambda n : format_to(float, n))(stocks.Open).cast('float'))
    stocks = stocks.withColumn('High',F.udf(lambda n : format_to(float, n))(stocks.High).cast('float'))
    stocks = stocks.withColumn('Low',F.udf(lambda n : format_to(float, n))(stocks.Low).cast('float'))
    #stocks.cache()

    stocks = stocks.dropna(subset = ['Date'])

    """   read the factors   """
    factor_columns = ['Date','Open','High','Low','Close','Adj Close','Volume','symbol']

    snp = sc.parallelize([l.strip() for l in open(f"{factors_dir}/GSPC.txt").readlines()])
    snp = snp.map(lambda l : l.split(sep = '\t')+['GSPC'])
    snp = spark.createDataFrame(snp.collect()[1:],factor_columns)

    snp = snp.withColumn('Date',F.udf(lambda datestr : parse_datestring(date_string = datestr, str_pattern = '%b %d, %Y'))(snp.Date).cast('date'))
    snp = snp.withColumn('Open',F.udf(lambda n : format_to(float, n))(snp.Open).cast('float'))
    snp = snp.withColumn('High',F.udf(lambda n : format_to(float, n))(snp.High).cast('float'))
    snp = snp.withColumn('Low',F.udf(lambda n : format_to(float, n))(snp.Low).cast('float'))
    snp = snp.withColumn('Volume',F.udf(lambda n : format_to(float, n))(snp.Volume).cast('float'))
    snp = snp.withColumn('Close',F.udf(lambda n : format_to(float, n))(snp.Close).cast('float'))
    snp = snp.withColumn('Adj Close',F.udf(lambda n : format_to(float, n))(F.col('Adj Close')).cast('float'))

    snp = snp.dropna(subset = ['Date'])

    nasdaq = pd.read_csv(f"{factors_dir}/IXIC.csv")
    nasdaq['symbol'] = 'IXIC'
    nasdaq = spark.createDataFrame(nasdaq,factor_columns)

    nasdaq = nasdaq.withColumn('Date',F.udf(lambda datestr : parse_datestring(date_string = datestr, str_pattern = '%Y-%m-%d'))(nasdaq.Date).cast('date'))
    nasdaq = nasdaq.withColumn('Open',F.udf(lambda n : format_to(float, n))(nasdaq.Open).cast('float'))
    nasdaq = nasdaq.withColumn('High',F.udf(lambda n : format_to(float, n))(nasdaq.High).cast('float'))
    nasdaq = nasdaq.withColumn('Low',F.udf(lambda n : format_to(float, n))(nasdaq.Low).cast('float'))
    nasdaq = nasdaq.withColumn('Volume',F.udf(lambda n : format_to(float, n))(nasdaq.Volume).cast('float'))
    nasdaq = nasdaq.withColumn('Close',F.udf(lambda n : format_to(float, n))(nasdaq.Close).cast('float'))
    nasdaq = nasdaq.withColumn('Adj Close',F.udf(lambda n : format_to(float, n))(F.col('Adj Close')).cast('float'))

    nasdaq = nasdaq.dropna(subset = ['Date'])

    bonds5y = pd.read_csv(f"{factors_dir}/FVX.csv")
    bonds5y['symbol'] = 'FVX'
    bonds5y = spark.createDataFrame(bonds5y,factor_columns)

    bonds5y = bonds5y.withColumn('Date',F.udf(lambda datestr : parse_datestring(date_string = datestr, str_pattern = '%Y-%m-%d'))(bonds5y.Date).cast('date'))
    bonds5y = bonds5y.withColumn('Open',F.udf(lambda n : format_to(float, n))(bonds5y.Open).cast('float'))
    bonds5y = bonds5y.withColumn('High',F.udf(lambda n : format_to(float, n))(bonds5y.High).cast('float'))
    bonds5y = bonds5y.withColumn('Low',F.udf(lambda n : format_to(float, n))(bonds5y.Low).cast('float'))
    bonds5y = bonds5y.withColumn('Volume',F.udf(lambda n : format_to(float, n))(bonds5y.Volume).cast('float'))
    bonds5y = bonds5y.withColumn('Close',F.udf(lambda n : format_to(float, n))(bonds5y.Close).cast('float'))
    bonds5y = bonds5y.withColumn('Adj Close',F.udf(lambda n : format_to(float, n))(F.col('Adj Close')).cast('float'))
    bonds5y = bonds5y.dropna(subset = ['Date'])

    bonds30y = pd.read_csv(f"{factors_dir}/TYX.csv")
    bonds30y['symbol'] = 'TYX'
    bonds30y = spark.createDataFrame(bonds30y,factor_columns)

    bonds30y = bonds30y.withColumn('Date',F.udf(lambda datestr : parse_datestring(date_string = datestr, str_pattern = '%Y-%m-%d'))(bonds30y.Date).cast('date'))
    bonds30y = bonds30y.withColumn('Open',F.udf(lambda n : format_to(float, n))(bonds30y.Open).cast('float'))
    bonds30y = bonds30y.withColumn('High',F.udf(lambda n : format_to(float, n))(bonds30y.High).cast('float'))
    bonds30y = bonds30y.withColumn('Low',F.udf(lambda n : format_to(float, n))(bonds30y.Low).cast('float'))
    bonds30y = bonds30y.withColumn('Volume',F.udf(lambda n : format_to(float, n))(bonds30y.Volume).cast('float'))
    bonds30y = bonds30y.withColumn('Close',F.udf(lambda n : format_to(float, n))(bonds30y.Close).cast('float'))
    bonds30y = bonds30y.withColumn('Adj Close',F.udf(lambda n : format_to(float, n))(F.col('Adj Close')).cast('float'))

    bonds30y = bonds30y.dropna(subset = ['Date'])

    
    factors_dict = dict({'snp': snp, 'nasdaq':nasdaq, 'bonds5y':bonds5y, 'bonds30y':bonds30y} )

    return stocks, factors_dict