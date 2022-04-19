# risk_assessment


Financial Risk Management with VaR analysis and prediction
    
 
 
Techniques followed

	VaR is a finacial statistic among others for assessing risk against an investment portfolio. It is often
	used before trading in financial markets, in order to help in making informed decisions. It is represented 
	by three parameters and it expresses the least loss in investment (l) that it is expected to happen with a
	probability (p) over a time horizon T. There are other similar finacial statistics for assessing risk when
	investing, such as CVaR, which is represented by three parameters, too, but in contast to VaR, it expresses
	the average investment loss that is expected to happen with probability p over a time horizon T. 	
	
	There have been multiple techniques suggested for the estimation of the VaR statistic, relying on carrying
	out analytics on estimated instrument returns. Some of the most popular are the Variance-Covariance method,
	the Historical Data and the Monte Carlo approach.
    
	The technique that is followed in this use case is based on Monte Carlo and follows the steps described in
	[3], deviating slightly at few points. Briefly described, it involves the execution of thousands simulations
	in parallel at multiple cores, for the aim of being able in that way to model market's behavior and 
	therefore, to get a reliable estimation of the instruments' returns, on which the VaR and cVaR statistics 
	will be estimated. 
	In these simulations, we assume that each trial runs a seperate market scenario, where each instrument's 
	return is determined by the market conditions, in which the probability ditribution of each instrument's 
	return over the market conditions, as well as the dynamics of the market conditions are not known, in fact, 
	a priori. The market conditions, for fascilitating quantitative analysis, are often expressed in certain 
	market factors, such as the S&P 500 Index 's perfomance, the Nasdaq 's returns, the changes in USD/EURO 
	exchange and others. The dynamics of these factors over time, for starting to build a solid ground on which
	we could estimate instruments' returns, in repsect with the changing market conditions, could be modeled 
	using generative modeling. There is a range of techniques that may be used at this point. We applied a model
	following a multivariate distribution of the factors' fluctuations over time. For modeling each instrument's 
	return with respect to factors' behavior, a linear (or a nonlinear) model may be used on market factors' 
	fluctuations (or on a transformation of the market factors' fluctuations if using a linear, instead of a non-
	linear instrument model). Each instrument model is then trained using historical data of the instrument 
	returns. The VaR will and the CVaR are computed on the p percentage of the worst trials' returns. 
	
	There are alternative simulation techniques for estimating Var and CVaR statistics, which differantiate on 
	the techniques proposed for modeling market factors and for estimating investment return based on these 
	market factors. The method applied here captures well a mainstream of the ways followed for estimating the 
	VaR and CVaR statistics and managing finacial risk.
	
	In another mindset, there are other methods for the calculation of VaR and other financial statistics, which
	are based on Deep Learning Networks (such as LSTMs and others).



Data set
   
    All the financial data used in this case study is retrieved from Finance Yahoo [1].
	
    -> historical data for the year 2013 for our portfolio of instruments, composed of all the stocks included
    in the NASDAQ index. It is composed of 2972 stocks and it includes around 600000 records, though the original 
    dataset downloaded consists of around 5856396 records, as it includes data from year 2000 up until 2013.
    Typically, for commercial applications far more data is used than this udes in this use case. The dataset 
    may be found in the folder 'data/stocks'.
    
    -> historical data for year 2013 about the market factors considered. It consists of about 1200 records for
    all the factors, in total. This dataset may be found in the folder 'data/factors'.
       4 factors considered in the analysis
	     ^GSPC     S&P500
		 ^IXIC     NASDAQ
		 ^TYX      prices of 30-year US Treasury bonds
		 ^FVX      prices of 5-year US Treasury bonds
	
    For downloading the historical data for the stocks you may use script [2], calling it from the working 
    directory. Ensure that the 'symbols.txt' file, with the symbols of the stocks, is in the working directory. 
	
	

Baseline

	The followed method of estimating VaR and CVaR statistics is aimed to be evaluated against other techniques
	in the future. Some indicative methods to use as benchmarks is the Variance-Covariance method, which assumes 
	that the investment return of each instrument follows normal distribution and can be modelled using the mean 
	and standard deviation from the historical data and the Modeling from Historical Data approach, which estimates
	VaR and CVaR from the historical data directly, considering the instruments' worst performance over a specific 
	time period.

	
	
Challenges
	
	Monte Carlo's performance is largely dependent on the reliability of the models chosen for the simulation of the
	market factors' behavior and of those modeling each instrument's return.
	
	In modeling instruments' behavior we have to face the training of a big number of linear models (around 3000) 
	with few features and relatively small amount of data points (historical data).
	
	Another challenge at a technical level encountered is the management of null values. Sometimes, in data preprocessing, 
	it can be tricky, when using different frameworks, as for example, pandas represents 'None' using 'pd.na', though
	this is not captured as 'null' value in pyspark, when using the column method 'isNull', which may lead to inconsistencies 
	and leakage of errors, if not taking appropriate actions in 'null' values's management, probably with a more significant 
	impact in series data processing.

     

Training process
    
	The instrument returns are modelled using linear models, trained each by minimizing empirical loss with Stochastic 
	Gradient Descent for 1000 iterations. There is definitely much room for improving the models. The data used for
	training each instrument model was around 200 for the time period of year 2013, which is considered very few. When 
	increasing the range of the time period, the linear model 's performance is again poor and affects the overall 
	portfolio's return that is estimated from the trials.

	An as much close to reality simulation of the instrument returns based on market conditions is attempted through 
	carrying out 10000 trials of instrument returns' estimation with 100 degree of parallelism. In each partition where
	trials are executed a different seed has been used for the random number generators, to achieve as much as possible 
	randomness in trials execution and approach a good empirical distribution estimation. 

    	

Evaluation

	In the method followed, first we need to know how much we can rely on this method for the computation of the VaR
	and CVaR statistics, as the model been used is composed of modules that exhibit uncertainty and variable accuracy 
	depended on the amount of historical data used and the number of trials and how these trials have been carried out.
	The confidence intervals for VaR and CVaR can provide us such an information and a nice way for estimating confidence 
	intervals is through bootstrapping [3]. Bootstrapping involves carrying out a number of times (e.g. 100 ) the computation
	of the statistics and then forming the confidence intervals for a specific probability from the corresponding quantiles 
	of the historical data. The confidence intervals thus give an estimate of how much consistent is the model in its output
	or how much confident the model is in its estimation of the statistic and in this case arise as very tiny, which is a 
	good indication for the quality of the model.	

	Secondly, we need to evaluate the statistics themeselves against the real world, how well they represent reality and 
	give the right signal about risk. Several tests have been proposed in the literature [4] for the evaluation of the VaR 
	statistic, which all of them have both strong and weak points. For this, it is a good practice to carry out a set of 
	them for the purpose of evaluating how good each statistic is against reality. In this study, Kupie's Portion-of-Failures 
	test [4, 5] and the binomial test have been carried out [4], which reveal that the value of the statistic computed doesn't 
	represent sufficiently reality, which means that some improvements in the model may be needed, such as training the 
	instrument models with more data or picking a better type of model for instrument returns' simulation.
	

 
Code

   risk_management.py
   
   
   The time window of 2 weeks for calculating the returns of investment that has been used in the implemenation is somewhat
   loose. In a later version of the code 10 business days will be considered for the computation of the historical returns 
   for achieving better quality of the historical data used in the modeling process.
   All can be run interactively with pyspark shell or by submitting  
       e.g. exec(open("project/location/risk_assessment/risk_management.py").read()) for an all at once execution. The code  
   has been tested on a Spark standalone cluster. For the Spark setting, spark-3.1.2-bin-hadoop2.7 bundle has been used.    
   The external python packages that are used in this implementation exist in the requirements.txt file. Install with:   
	   pip install -r project/location/risk_assessment/requirements.txt
     


References

	1. https://finance.yahoo.com/
	2. https://github.com/sryza/aas/blob/master/ch09-risk/data/download-all-symbols.sh
	3. Advanced Analytics with Spark, Sandy Ryza, Uri Laserson, Sean Owen, & Josh Wills
	4. https://www.mathworks.com/help/risk/overview-of-var-backtesting.html
	5. Kupiec, P. "Techniques for Verifying the Accuracy of Risk Management Models." Journal of Derivatives. Vol. 3, 1995, pp. 73–84.
	
	
