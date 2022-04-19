from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split	
import warnings

def linear_regressor(stock, x, y):       #   linear model fitted by minimizing a regularized empirical loss with SGD
    dim = min(len(y), x.shape[0])
    
    y = y[:dim]
    x = x.iloc[:dim,:]	                                   
    #print(f"stock = {stock}")
    #sgdr = ElasticNet(tol=1e-3, max_iter=100)
    sgdr = SGDRegressor(tol=1e-3, max_iter=1000, shuffle = False)
    lr = LinearRegression()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model = sgdr.fit(x,y)
        #model = lr.fit(x,y)
        
    return model
