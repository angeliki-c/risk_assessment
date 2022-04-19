
import os
import sys
import warnings
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
# in case the module was not found by python, append to $PYTHONPATH the dir of the 
# project 
# sys.path.append('/risk_assessment/')

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, FloatType, DateType, IntegerType, StructType, TimestampType
import pyspark.sql.functions as F  

from risk_assessment.load_the_data import  load_the_data
from risk_assessment.utils import days, fillnulls_with_recent
from risk_assessment.models import linear_regressor
from risk_assessment.sampling import sample_data_series, sample_from_multivariate_dist, sample_data_rdd
from risk_assessment.carry_out_trials import *


start_date = "2013-01-01"            #  provide a start date based on the data available in the format year-month-day
end_date = "2013-12-31"
current_working_dir = ""

if current_working_dir != "":
    os.chdir(current_working_dir)
else:
    print(f'\nYour current working dir is : {os.getcwd()}. Ensure that the project is in this dir.\n')

sc.setLogLevel('OFF')
spark = SparkSession.builder.appName('appName').getOrCreate()
SparkContext.setSystemProperty('spark.executor.memory','4g')
SparkContext.setSystemProperty('spark.driver.memory','4g')
SparkContext.setSystemProperty('spark.master','local[*]')
SparkContext.setSystemProperty("spark.scheduler.mode", "FAIR")
#SparkContext.setSystemProperty('spark.executor.cores', '8')

                                                               

#   Load the data

stocks_raw, factors_dict_raw = load_the_data(spark)

#   Compute stock returns and factor fluctuations across a time interval of 2 weeks for the time period selected 

print(f"\nCompute the stock returns and the factor fluctuations per 2 weeks for thetime period  {start_date} - {end_date}...")
# Filter the data keeping those corresponding to the time period start_date - end_date 
stocks = stocks_raw.where(f'Date >= date("{start_date}") and Date <= date("{end_date}") ').select(['Date','Open','symbol'])
stocks = stocks.dropDuplicates()
factors_dict = dict()
for name, df in factors_dict_raw.items():
    factors_dict[name] = df.where(f'Date >= date("{start_date}") and Date <= date("{end_date}") ').select(['Date','Open'])
    factors_dict[name] = factors_dict[name].dropDuplicates()
    
# Ensure that the historical data is sorted from the most recent to the oldest.  
stocks = stocks.withColumn('Date', stocks.Date.cast(TimestampType()).cast('long'))
# Store it efficintly in the partitions to reduce the number of reshuffle operation in the following computations.
stocks = stocks.repartition(stocks.symbol).sortWithinPartitions(stocks.symbol, stocks.Date, ascending = False) 	
for name, df in factors_dict.items():
    factors_dict[name] = df.sort('Date',ascending = False)
    
    
# This data set has several null values, which will be replaced by the most recent
# values. 
# bonds5y = bonds5y.fillna(method = 'ffill')
# Unfortunately 'fillna' method in pyspark DataFrame has a bug at the time (2022-03)
# Alternatively, for avoiding a more complex solution using 'joins' between dataframes, we are 
# converting the pyspark dataframe to a pandas dataframe and we are using pandas 'apply' method.
stockspd = stocks.toPandas()
stockspd['Open'] = stockspd.groupby(by = 'symbol')['Open'].apply(fillnulls_with_recent)
stocks = spark.createDataFrame(stockspd)

for name, df in factors_dict.items():
    dfpd = df.toPandas()
    dfpd['Open'] = fillnulls_with_recent(dfpd['Open'])
    df = spark.createDataFrame(dfpd)
    factors_dict[name] = df

for name, df in factors_dict.items():
    df = df.withColumn('Date', df.Date.cast(TimestampType()).cast('long'))
    factors_dict[name] = df

def compute_returns(r):
    end_date = r.dates[0]
    start_date = r.dates[1]
    if end_date - start_date > timedelta(10).total_seconds():
        return round((r.values[0] - max(r.values[1],0.00000001))/max(r.values[1],0.00000001),3)
    else: 
        return None
        
    
from pyspark.sql import Window
# The Window class in pyspark considers the order specified with 'orderBy' in creating the window 
# boundries. You have only to take care of its width and the starting point.
window_spec = Window.partitionBy('symbol').orderBy(F.desc('Date')).rangeBetween(0,days(14))
stock_returns = stocks.withColumn('fluct', F.udf(lambda s : compute_returns(s))(F.struct(F.array(F.first(stocks.Date).over(window_spec) , F.last(stocks.Date).over(window_spec)).alias('dates'),F.array(F.first(stocks.Open).over(window_spec).cast('float') , F.last(stocks.Open).over(window_spec).cast('float')).alias('values')))).where('fluct is not null')
#stock_returns = stock_returns.withColumn('window', F.array(F.first(F.col('Date')).over(window_spec).cast(TimestampType()).cast('date') , F.last(F.col('Date')).over(window_spec).cast(TimestampType()).cast('date')))
stocks = stocks.withColumn('Date', stocks.Date.cast(TimestampType()).cast('date'))
stock_returns = stock_returns.withColumn('Date', stock_returns.Date.cast(TimestampType()).cast('date'))

window_spec_f = Window.orderBy(F.desc('Date')).rangeBetween(0, days(14))
factors_fluct_dict = dict()
for name, df in factors_dict.items():
    df_fluct = df.withColumn('fluct', F.udf(lambda s : compute_returns(s))(F.struct(F.array(F.first(F.col('Date')).over(window_spec_f) , F.last(F.col('Date')).over(window_spec_f)).alias('dates'),F.array(F.first(F.col('Open')).over(window_spec_f).cast('float') , F.last(F.col('Open')).over(window_spec_f).cast('float')).alias('values')))).where('fluct is not null')
    #df_fluct = df_fluct.withColumn('window', F.array(F.first(F.col('Date')).over(window_spec_f).cast(TimestampType()).cast('date') , F.last(F.col('Date')).over(window_spec_f).cast(TimestampType()).cast('date')))
    factors_dict[name] = df.withColumn('Date', df.Date.cast(TimestampType()).cast('date'))
    df_fluct = df_fluct.withColumn('Date', df_fluct.Date.cast(TimestampType()).cast('date'))
    df_fluct = df_fluct.withColumn('fluct', df_fluct.fluct.cast('float'))
    factors_fluct_dict[name] = df_fluct
    

print(f"\nStatistics on the average return of the investment portfolio for the time period {start_date} - {end_date} :")
stock_returns.groupBy('Date').agg(F.avg('fluct').alias('avg_return')).describe().show()
for name, df in factors_fluct_dict.items():
    print(f"\nStatistics on the {name.upper()} fluctuations for the time period {start_date} - {end_date} :")
    factors_fluct_dict[name].select(['Date','Open','fluct']).describe().show()


from pyspark.sql import Row
def _preprocess(fluct):
    squared = np.sign(fluct) * np.power(fluct,2)
    #powered = np.power(fluct,3)
    root_squared = np.sign(fluct) * np.sqrt(np.absolute(fluct))
    feature = float(squared + root_squared + fluct)
    
    return feature
	
# the first dimension of the feature set
dim_0 = np.inf    
for name, df in factors_fluct_dict.items():
    size = df.count()
    if size < dim_0:
        dim_0 = size

# the features for learning the models are non-linear functions of the factor fluctuations observed on a per-2-week basis
features = spark.range(dim_0)
features = features.hint('broadcast') 

ws = Window.orderBy(F.desc('Date')).rowsBetween( Window.unboundedPreceding,Window.currentRow)
wss = Window.partitionBy('symbol').orderBy(F.desc('Date')).rowsBetween( Window.unboundedPreceding,Window.currentRow)
# featurization : transformation of factor flactuations
for name, df in factors_fluct_dict.items():
    rdd = df.limit(dim_0).rdd.map(lambda r : Row( _preprocess(r.fluct), r.Date))
    df = spark.createDataFrame(rdd,StructType().add(f'{name}_feat', FloatType(), True).add('Date', DateType(),True), [f'{name}_feat', 'Date'])
    df = df.withColumn('id', F.row_number().over(ws) - 1)
    df = df.drop('Date')
    features = features.join(df, 'id')
    
feat_cols = [ col for col in features.columns if col not in ['id', 'Date']]

#features_pd = features.toPandas()
#features_br = sc.broadcast(features_pd)	

def split_without_shuffle(train_ratio, df):
    df_sizes = df.groupBy('symbol').agg(F.max('id').alias('max_id'))
    train_df = df.join(df_sizes, on = 'symbol').where(f'id <= (0.8 * max_id)')
    test_df = df.join(df_sizes, on = 'symbol').where(f'id > (0.8 *  max_id)')
    return train_df, test_df
    
#    Create sdf_train and sdf_test spark dataframes for training and testing the stock_return models    

sdf = stock_returns.selectExpr(['symbol','Date','cast(fluct as float) as label'])
sdf = sdf.withColumn('id', F.row_number().over(wss) - 1)
sdf  = sdf.join(features, 'id')
sdf = sdf.drop('Date')
sdf_cols = sdf.columns 
sdf_train, sdf_test = split_without_shuffle(0.8, sdf)
stock_returns_rdd = sdf.selectExpr(sdf_cols).rdd.map(lambda r : (r.symbol,r))
stock_returns_rdd = stock_returns_rdd.groupByKey().mapValues(pd.DataFrame)
stock_returns_rdd = stock_returns_rdd.filter(lambda tpl : tpl[1].empty == False)
stock_returns_rdd = stock_returns_rdd.map(lambda tpl : (tpl[0], tpl[1].rename({i:col for i,col in enumerate(sdf_cols)}, axis = 1)))
print("\nTrain the models....")
stock_models = stock_returns_rdd.map(lambda tpl : (tpl[0], linear_regressor(tpl[0],tpl[1][feat_cols],  tpl[1]['label'])))
#stock_models = stock_returns_rdd.map(lambda tpl : (tpl[0], linear_regressor(tpl[0],features_br.value,  tpl[1]['label'])))
stockmodels_weights = stock_models.map(lambda m : (m[0], m[1].coef_))

stock_returns_rdd_test = sdf_test.selectExpr(sdf_cols).rdd.map(lambda r : (r.symbol,r))
stock_returns_rdd_test = stock_returns_rdd_test.groupByKey().mapValues(pd.DataFrame)
stock_returns_rdd_test = stock_returns_rdd_test.filter(lambda tpl : tpl[1].empty == False)
stock_returns_rdd_test = stock_returns_rdd_test.map(lambda tpl : (tpl[0], tpl[1].rename({i:col for i,col in enumerate(sdf_cols)}, axis = 1)))
joined_rdd1 = stock_returns_rdd_test.join(stock_models)
print("\nCalculate the predictions....")
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    predictions_rdd = joined_rdd1.map(lambda tpl : (tpl[0], tpl[1][1].predict(tpl[1][0][feat_cols])))
    predictions_rdd.cache()
    
joined_rdd2 = predictions_rdd.join(stock_returns_rdd_test)

#    Compute r2 scores    
print("\nCompute the r2 scores (coefficient of determination) to evaluate the linear models trained.... ")
from sklearn.metrics import r2_score 
r2_scores_rdd = joined_rdd2.map(lambda tpl : (tpl[0], float(r2_score(tpl[1][1]['label'], tpl[1][0]))))
r2_scores = spark.createDataFrame(r2_scores_rdd,StructType().add('symbol', StringType(), True).add('score', FloatType(), True), ['symbol','score'])
r2_scores.cache()


print("\nShow statistics on the r2 scores....")
r2c = r2_scores.rdd.filter(lambda el : (pd.isna(el.score) == False) & (el.score != None)).toDF(['symbol','score'])
r2c.describe().show()

#   Data Sampling 

dim = np.inf
density_dict = dict()
factors_pdf = pd.DataFrame()
domain_dict = dict()

print("""\n R2 is not expected to evaluate in a reliable way the models in case that the 'test' data set for each instrument is small,
 and if the features are correlated (from practice we know that some market factors are correlated). """)     


factors_fluct_pdfs = dict()
for k,sdf in factors_fluct_dict.items():
    #print(k)
    pdf = sdf.toPandas()
    factors_fluct_pdfs[k] = pdf
    print("\nApplying kernel density estimation for gaining an insight of the underlying distribution of each factor's returns...")
    domain_dict[k], density_dict[k] = sample_data_series(pdf['fluct'],k)
    if dim > pdf.shape[0]:
        dim = pdf.shape[0]
  
  
for k,pdf in factors_fluct_pdfs.items():
    factors_pdf[k]  = pdf['fluct'][:dim]



print("\nThe factor returns based on the historical data follow a distribution that resembles the normal distribution. ")
print("""\nLets investigate the corelation that exists between the factors. From practice, we know that exists some 
      correlation between the factors e.g. between and s&p and nasdaq. Therefore for simulating the market based on
      factor fluctuations we should take this correlation into consideration. Instead of modelling the market using 
      independent normal distributions we 'd rather model it using multivariate normal distribution with a non-diagonal
      covariance matrix.""")    
 
pearsons = factors_pdf.corr()    
print(f"\nPearson's correlations between factor fluctuations : \n{pearsons}")  

#   Sample data from a multivariate distribution for the factor values.
print("""\nThe multivariate distribution for simulating the market will be constructed using the 'Cholesky' decorelation method. 
         For fitting the model the sample means and the covariance matrix of the historical factor fluctuations will be used.""")    
         
sample = sample_from_multivariate_dist(factors_pdf)

stock_models_bc = sc.broadcast(stock_models.collect())
factor_means = factors_pdf.mean(axis = 0).to_numpy()
factor_covs = factors_pdf.cov().to_numpy()

scaffold = dict({'factor_means': factor_means, 'factor_covs' : factor_covs, 'feat_cols' : feat_cols, 'stock_models_val':stock_models_bc.value})

trials_df, trials_rdd = trials(spark, parallelism = 100, num_trials = 10000, scaffold = scaffold)
    

def compute_var(df, p = 0.05):
    #value_at_risk = df.quantile([0.05])['average_return'][0.05]
    quantiles = df.stat.approxQuantile('average_return',[p],0.0)
    value_at_risk = quantiles[0]
    return value_at_risk
    
    
value_at_risk = compute_var(trials_df)
print(f"""\nVar is estimated at {value_at_risk}. This means that there is 5% probability of loosing at least {value_at_risk} degree of the investment over the
        last 2-week period.""")

def compute_cvar(df):
    var = compute_var(df)
    avr_ret_values = [r.average_return for r in df.where(df.average_return <= var).collect()]
    conditional_value_at_risk = sum(avr_ret_values)/len(avr_ret_values)

    return conditional_value_at_risk


conditional_value_at_risk = compute_cvar(trials_df)
print(f"""\nCVar is estimated at {conditional_value_at_risk}. This means that there is 5% probability of loosing {conditional_value_at_risk} degree of the investment over the
        last 2-week period.""")

def get_confidence_intervals(dataset, num_sampling = 100, statistic = None, probability = None):
    # the size of sample equals the size of the dataset
    gen = default_rng(1)
    
    values = []
    for i in range(num_sampling):
        random_state = gen.integers(low = 0, high =  2**32 - 1, size = 1)[0]
        sample = dataset.sample(True, 1.0, random_state )
        #dpf = dataset.toPandas()
        #sample = dpf.sample(n = num_sampling, random_state = random_state, axis = 0, replace = True)
        #print(f"sample = {sample}")
        #sample.show()
        stat = statistic(sample)
        #print(stat)
        values = values + [stat]
    sorted_values = sorted(values)
    [lower, upper] =  [sorted_values[int((num_sampling * probability/2) -1)], sorted_values[int(num_sampling * (1 - probability/2))]]    
    return [lower, upper]
        
# Evaluation of the statistics' model.
probability = 0.05


confidence_interval_var = get_confidence_intervals(trials_df, num_sampling = 100, statistic = compute_var, probability = probability)
confidence_interval_cvar = get_confidence_intervals(trials_df, num_sampling = 100, statistic = compute_cvar, probability = probability)

print("\nEvaluation of the VaR and CVaR models...")
print("""\nCarrying out bootstrapping, an empirical distribution is formed for each statistic. The confidence intervals will be 
       computed for each one as an estimation of the confidence the model has in estimating right each statistic.""")
print(f"\nThe confidence interval for the VaR at p-level is {confidence_interval_var}. ")
print(f"\nThe confidence interval for the CVaR at p-level is {confidence_interval_cvar}. ")

print("\nEvaluation of the VaR statistic against reality...")
print("\nCarrying out Kupiec's Proportion of Failures test... ")

def compute_failures(sr, risk_value):
    st_ret_list = sr.groupBy('symbol').agg(F.sum(F.col('fluct')).alias('total_return'), F.count(F.col('Date')).alias('cnt')).collect()
    failures = 0
    size = 0
    for r in st_ret_list:
        ret = r.total_return/r.cnt
        if (ret < 0) & (ret < risk_value):
            failures += 1
        size += r.cnt
        
    return failures, size 
        
        
failures, size = compute_failures(stock_returns, value_at_risk)        
failures2, size2 = compute_failures(stock_returns, conditional_value_at_risk) 
# the 'conf_level' argument accepts the confidence level for the VaR statistic
def compute_kupiec_pof_statistic(failures, p_value, size):
    stat = -2 *((size - failures) * np.log(1 - p_value) + failures * np.log(p_value)- (size - failures) * np.log(1 - failures/size) - failures * np.log(failures/size))
    return stat


kupiec_pof_stat = compute_kupiec_pof_statistic(failures, 0.05, size)

print("""\n The Kupiec 's POF statistic is asymptotically distributed as a chi-square variable with 1 degree of freedom [4]. If this statistic exceeds the critical 
      of the chi-square statistic we have much evidence to reject the null hypothesis that the VaR statistic is reasonable against the observed data.
      Whereas the confidence interval showed that the model is quite consistent in estimating the VaR statistic, the test suggests that the VaR model 
      doesn't represent well enough the observed world.\n""")    

print(f"""\nThe Kupiec POF statistic for VaR is equal to {kupiec_pof_stat}. The corresponding p-value to the same statistic value in the Chi Squared 
      distribution should be examined in order to decide whether there is sufficient evidence to reject the null hypothesis. In this case study it was
      found much smaller than the p-value of the VaR statistic.        """)
      
from scipy.stats import chi2
print(f"""\nThe probability density function of a chi squared ditribution at the value of the Kupiec's statistic is : {chi2.pdf(kupiec_pof_stat,df = 1)}. 
       If this is much lower than {probability} the null hypothesis that our VaR model is reasonable (conforms to the observations) is rejected.""")    
print("\nEvaluation of the VaR statistic against reality with the binomial test... ")

    
# the expected number of failures is approximated by the sample mean failures/size
def compute_binomial_statistic(failures, p_value, size):
    stat = (failures - size * p_value)/(np.sqrt(size * p_value * (1 - p_value)))
    return stat

binomial_stat = compute_binomial_statistic(failures, 0.05, size)


print("""\nThe binomial statistic is asymptotically distributed as a standard normal distribustion [4]. If this statistic exceeds the critical value
      of the standard normal distribution statistic, we have much evidence to reject the null hypothesis that the VaR statistic is reasonable against
      the observed data. Whereas the confidence interval showed that the model is quite consistent in estimating the VaR statistic, the test suggests
      that the VaR model doesn't represent well enough the observed world.\n""")    

print(f"""\nThe binomial statistic for VaR is equal to {binomial_stat}. The corresponding p-value to the same statistic value in the Standard Normal 
      distribution should be examined in order to decide, whether there is sufficient evidence to reject the null hypothesis. In this case study it was
      found much smaller than the p-value of the VaR statistic.        """)


    


print("""\nApplying kernel density estimation for gaining an insight on the underlying distribution of the average portfolio's 
return estimated after the execution of the simulations ...""")

min_val, max_val, count, stddev = trials_df.agg(F.min('average_return').alias('min'), F.max('average_return').alias('max'), F.count('average_return').alias('count'), F.stddev_pop('average_return').alias('stddev')).collect()[0]
sample_data_rdd(trials_rdd, min_val, max_val, count, stddev)

