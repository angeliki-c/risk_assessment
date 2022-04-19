from scipy.stats import multivariate_normal
from numpy.random import default_rng
import numpy as np
import pandas as pd

from pyspark.sql.types import FloatType, StructType

from risk_assessment.utils import average


def preprocess_sample(sample, feat_cols):
    
    feature = sample +  np.sign(sample) * np.power(sample, 2) + np.sign(sample) * np.sqrt(np.absolute(sample))
    feature = pd.DataFrame(feature.reshape(1,4), columns = feat_cols)
    return feature
    
    

def trial_returns(seed, num, feat_cols, factor_means, factor_covs, models):
    trial_returns = [0]*num
   
    gen = default_rng(seed)     #  create a random number generator
    #bit_generator = MT19937(seed)                  # Mersenne Twister generator of random numbers
    #rn = bit_generator.state['state']['key'][:factor_covs.shape[0]]
    
    for i in range(num):
        sample = multivariate_normal.rvs(mean = factor_means, cov = factor_covs, random_state = gen.integers(low = 0, high =  2**32 - 1, size = 1)[0])
        #sample = sample_from_multivariate_dist(factors_pdf)
        feature = preprocess_sample(sample, feat_cols)
        
        #print(f"trial_return {i}  ")
        trial_returns[i] = trial_return(feature,models)
        
    #print("trial_returnS done")    
    return list(trial_returns) 


def trial_return(factor_features, models):
    size = 0
    total_return = 0
    #print(f"{factor_features}\n")
    for t in models:
        stock_return_prediction = t[1].predict(factor_features)
        #print(f"{ t[1].coef_.reshape((1,4))}\n")
        #stock_return_prediction = np.dot(t[1].coef_.reshape((1,4)), factor_features.values.T)
        size = size + 1 
       
        total_return = total_return + stock_return_prediction
        #print(stock_return_prediction)
    #print('Done for all instr')    
    avg_instr_return = float(total_return/size)
    return avg_instr_return
    
    
    
def trials(spark, parallelism = None, num_trials = None, scaffold = None,  base_seed = 1259)    :
    sc = spark.sparkContext
    parallelism = parallelism
      
    seeds_list = range(base_seed, base_seed + parallelism)
    seeds = sc.parallelize(seeds_list,parallelism)
    num_trials = num_trials
   
    factor_means = scaffold['factor_means']
    factor_covs = scaffold['factor_covs']
    feat_cols = scaffold['feat_cols']
    stock_models_val = scaffold['stock_models_val']
  
    trials = seeds.flatMap(lambda s : trial_returns(s, num_trials//parallelism, feat_cols, factor_means, factor_covs, stock_models_val))
    trials.cache()

    schema = StructType().add('average_return',"float")
    trials_df = spark.createDataFrame(trials, FloatType())
    trials_df = trials_df.withColumnRenamed('value','average_return')

    return trials_df, trials