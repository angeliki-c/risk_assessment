import matplotlib.pyplot as plt


import pandas as pd
import numpy as np

import os

def sample_data_series(ser,k):
    from sklearn.neighbors import KernelDensity
    data = ser.to_numpy()
    max_val = max(data)
    min_val = min(data)
    stddev = ser.std()
    bandwidth = 1.06 * stddev * np.power(ser.shape[0], -0.2)
    #bandwidth = 0.02
    domain = np.arange(min_val, max_val , (max_val - min_val)/100 )
    kde = KernelDensity(kernel = 'gaussian', bandwidth = bandwidth)
    kdm = kde.fit(data.reshape(data.shape[0],1))
    density = kdm.score_samples(domain.reshape(domain.shape[0],1))
    fig = plt.figure(figsize = (8,8))
    plt.plot(domain, density, label = k)
    plt.legend()
    plt.xlabel("Returns")
    plt.ylabel("Density")
    plt.savefig(f"{os.getcwd()}/risk_assessment/{k}_density.png")
    print(f"\nThe estimated distribution (probability density function) for {k} is depicted on an image located at : {os.getcwd()}/risk_assessment/{k}_density.png")
    plt.clf()
    
    return domain,density

def sample_data_rdd(rdd, min_val, max_val, count, stddev):
    from pyspark.mllib.stat import KernelDensity
    import matplotlib.pyplot as plt
    bandwidth = 1.06 * stddev * np.power(count, -0.2)
    domain = np.arange(min_val, max_val , (max_val - min_val)/100 )
    kde = KernelDensity()
    kde.setBandwidth(bandwidth)
    kde.setSample(rdd)
    densities = kde.estimate(domain)
    fig = plt.figure(figsize = (8,8))
    plt.plot(domain, densities)
    plt.savefig(f"{os.getcwd()}/risk_assessment/returns_dist.png")
    plt.clf()    
    print(f"The estimated distribution is depicted on an image stored in {os.getcwd()}/risk_assessment/returns_dist.png ")
     
 
 
def sample_from_multivariate_dist(factors_pdf):
    mean = factors_pdf.mean(axis = 0).to_numpy()      # across factors
    cov = factors_pdf.cov().to_numpy()
    x = np.random.default_rng().multivariate_normal(mean, cov, size=mean.shape[0], method = 'cholesky')
    
    return x
    