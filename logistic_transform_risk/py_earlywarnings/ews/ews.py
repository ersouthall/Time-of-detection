import numpy as np
import pandas as pd
from . import kolmogorov_complexity
from . import entropy

from scipy.stats import iqr

def get_ews(x, windowsize, ac_lag, se = True, kc = True, mv_method="uniform",
            detrend_prev = False, original_data=None, center = False):
    '''
    Calculate early-warning signals from time series data.

    INPUT
    :param x: time series data (either 1-dimensional numpy array or list)
    :param windowsize: integer number of time points in moving window
    :param ac_lag: integer lag used to calculated the autocovariance, autocorrelation and decay time
    :param se: Boolean. If True calculate the moving Shannon entropy. This is significantly slower
    :param kc: Boolean. If True calculate the moving Kolmogorov complexity. This is around 10x slower
    :param mv_method, can be "uniform", "gaussian" or "exp"
    :param detrend_pre. If True the data is already detrended and no detrending will be performed in this function.
    :param original_data. The un-detrended data can be inputted and used for the calculation of the mean
    :param center. For the moving window calculations, the calculation can either be done center = True or for right-centered window averages (center = False)
    RETURN
    : dict containing all early-warning signals and original time series
    '''

    w = windowsize

    y = pd.Series(x, dtype = 'float')


    out = pd.DataFrame({"timeseries": y})
    if detrend_prev:
        out["mean"] = original_data
        y_detrend = y
    else:
        out["mean"] = mvw(y, w, mv_method, center = True)
        y_detrend = y-out['mean']
    if mv_method == 'gaussian':
        mv_method_else = "uniform"
        # w_else = 2*w
    else:
        mv_method_else = mv_method

    auto_res = autocov(y= y, w= w, lag = ac_lag)
    auto_res2 = autocov(y=y, w=w, lag = 2)
    out["autocovariance"] = auto_res['cov']
    out['autocorrelation'] = auto_res['corr']

    out['ac2'] = auto_res2['corr']
    out['acov2'] = auto_res2['cov']
    out["variance"] = mvw((y_detrend) ** 2, w,
                        mv_method_else, center = center)  
    out["standard_deviation"] = np.sqrt(out["variance"])

    out["coefficient_of_variation"] = out["standard_deviation"]/out["mean"]
    out["index_of_dispersion"] = out["variance"]/out["mean"]
    out["skewness"] = mvw((y_detrend) ** 3, w, mv_method_else, center = center) / (
        out["variance"] ** (3 / 2))
    out["kurtosis"] = mvw((y_detrend) ** 4, w, mv_method_else, center = center) / (
    out["variance"] ** (2))
    out["decay_time"] = -ac_lag / np.log(np.abs(out["autocorrelation"]))
    mu = out["mean"]


    # Kolmogorov complexity:
    if(kc == True):
        out["Kolmogorov_complexity"] = kolmogorov_complexity.CMovingKC(x,mu, windowsize)
        
    # Shannon entropy
    if(se == True):
        with np.errstate(divide='ignore', invalid='ignore'):
            out["Shannon_entropy"] = entropy.MovingEntropy(x, windowsize)


    return(out)

def silvermans(x):
    '''
    Implement Silvermans Rule of Thumb
    '''
    return 0.9*np.minimum(np.std(x), iqr(x)/1.34)*((len(x))**(-1/5))

def mvw(x, w, weight="exp", center = True):
    '''
    Function to determine how the moving window EWSs statistics should be calculated
    INPUTS:
    x: Pandas Series. Time-series data 
    w: Integer. Window size to perform window detrending on
    weight: Str. Weighting to use for calculations (i.e. all equal weight="uniform"; gaussian weight weight = "gaussian" or exponential weight="exp")
    center: Boolean. If True then calculations will be done an a center window. If False then calculations done on a right-window
    '''
    if weight == "exp":
        return x.ewm(halflife=w).mean()
    if weight == "uniform":
        return x.rolling(w, center = center).mean()
    if weight == 'gaussian':
        std_val = silvermans(x)
        return x.rolling(center = True, 
                            window = w,
                            min_periods=1,
                            win_type = weight).mean(std = std_val)
        
def autocov(y, w, lag = 1):
    '''
    function for calculating the autocovariance and autocorrelation 
    INPUTS: 
    y:  Pandas Series. Time-series data 
    w: Integer. Window size to calculate EWS on
    lag: Integer. Lag for the autocorrelation (e.g. AC(1) for lag-1)
    '''
    y_shift = y.shift(lag)
    y_shift = y_shift.rename('shift')
    y = y.rename('original')
    df = pd.concat([y, y_shift], axis =1)
    return_dict = {'cov':df.rolling(window = w).cov(pairwise = False)['original'],
                    'corr':df['original'].rolling(center = False,
                                            window = w).corr(df['shift'])
                    }
    return return_dict