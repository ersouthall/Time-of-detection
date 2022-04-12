import numpy as np
import pandas as pd 
from scipy.io import loadmat
from scipy.stats import skew, kurtosis, kendalltau
from itertools import combinations, groupby
from scipy.ndimage.filters import gaussian_filter as gf

def autocorrelation_tau(detrended_data, param_dict):
    '''
    param_dict requires 
        repeats (or number of places)
        T (total time period)
        BT (set to zero)
        windowSize (for the calculating )
        lag_tau '''
    AC_results = np.zeros((param_dict['repeats'], param_dict['T']+param_dict['BT']))
#     wd = param_dict['windowSize']//2
    wd = round(param_dict['windowSize']/2)
    for i in (range(wd,param_dict['T']+param_dict['BT']-(wd))):
        if param_dict['windowSize']%2==0:
            subtract = 1
        else:
            subtract = 2
        data_window = detrended_data[:,( i-(wd)):(i+(wd))]
        x = data_window[:, :-1] -np.transpose((np.nanmean(data_window[:,:-param_dict['lag_tau']],
                                                 axis = 1),)*(param_dict['windowSize']-subtract
                                                                ))
        x_tau = data_window[:,1:] -  np.transpose((np.nanmean(data_window[:,param_dict['lag_tau']:],
                                                        axis = 1),)*(param_dict['windowSize']-subtract
                                                                ))
        numerator = np.mean(x*x_tau, axis = 1)
        denominator = np.std(x,axis =1)*np.std(x_tau, axis = 1)
        xtest_autocorr=numerator/denominator
        # below is a different method (using inbuilt Pandas series function) but takes longer
    #     xtest_autocorr = pd.DataFrame(xtest.T).apply(lambda x:  pd.Series(x).autocorr(lag = 1)).values 
        AC_results[:,i] = xtest_autocorr
    return AC_results 


def populationEstimation(pop_2015, years, rate):
    sign = lambda x: (1, -1)[x<0]
    a_de_ppreciation = years - 2015
    growth = [pop_2015*(1+sign(value)*rate)**abs(value) for value in a_de_ppreciation ]
    return growth
    
def no_detrending(df):
    return df

def realisation_detrending(df):
    remove_mean_space = df.sub(df.mean(axis = 1), axis = 'rows')
    return remove_mean_space

def decay_time(AC):
    denom = np.log(np.minimum(np.maximum(AC, 0), 1))
    return -1/denom

def pandas_zscore(dataframe):
    rolling_mean = dataframe.expanding().mean()
    rolling_std = dataframe.expanding().std(ddof=0)
    normalise_stats = (dataframe - rolling_mean)/rolling_std
    return normalise_stats

def twosigma(standardised):
    n = len(standardised)
    running_average = np.zeros(n)
    running_std= np.zeros(n)
    for j in range(n):
        running_average[j] = np.nanmean(standardised[1:j+1])
        running_std[j] = np.nanstd(standardised[1:j+1])

        
    return running_std, running_average

def statistics_on_window(dataframe, window_size, detrend_function, window_detrend = None, std = None, filtering_size = 4, center = True):
#     print(detrend_function)
    if detrend_function == linear_window_detrending:

        detrended_data = detrend_function(df = dataframe, 
                                          BW = filtering_size ,
                             wind = str(window_detrend), 
                                          std_val=std)
    elif detrend_function == gf:
        smooth_data = np.zeros((dataframe.shape[0], dataframe.shape[1]))
        for index_col, col in enumerate(dataframe.columns):
            data = dataframe[col].values
            smooth = gf(data, sigma=std, mode='reflect')
            smooth_data[:, index_col] = smooth
        detrended_data = dataframe - pd.DataFrame(data = smooth_data)
    else:
        detrended_data = detrend_function(dataframe)
    statistics_window ={}
    statistics_window['Variance'] = detrended_data.rolling(center = center,
                             window = window_size,
                             win_type=None,
                             min_periods=1
                            ).var()
    statistics_window['Mean'] = detrended_data.rolling(center = center,
                                 window = window_size,
                                 win_type=None,
                                 min_periods=1
                                ).mean()
    statistics_window['First difference'] =(detrended_data.rolling(center = center,
                                 window = window_size,
                                 win_type=None,
                                 min_periods=1
                                ).var()).diff(axis = 0)
    statistics_window['Index of dispersion'] = detrended_data.rolling(center = center,
                             window = window_size,
                             win_type=None,
                             min_periods=1
                            ).var()/dataframe

    statistics_window['CV'] = detrended_data.rolling(center = center,
                             window = window_size,
                             win_type=None,
                             min_periods=1
                            ).std()/dataframe

    statistics_window['Kurtosis'] = detrended_data.rolling(center = center,
                                 window = window_size,
                                 win_type=None,
                                 min_periods=1
                                ).kurt()
    statistics_window['Skewness'] = detrended_data.rolling(center = center,
                                 window = window_size,
                                 win_type=None,
                                 min_periods=1
                                ).skew()


    ### autocorrelation 
    matrix_time_place = detrended_data.to_numpy()
    matrix_place_time = matrix_time_place.T
    ac_tau_1 = autocorrelation_tau(matrix_place_time, param_dict={'repeats': len(dataframe.columns),
                                                             'T': len(dataframe.index),
                                                             'BT': 0,
                                                             'windowSize': window_size,
                                                             'lag_tau': 1})

    statistics_window['AC(1)'] =pd.DataFrame(ac_tau_1.T, index=dataframe.index, columns= dataframe.columns)
    statistics_window['Decay Time'] = pd.DataFrame(decay_time(ac_tau_1).T,
                                                   index=dataframe.index, 
                                                   columns= dataframe.columns)
    statistics_window['CV'][statistics_window['CV']==np.inf] = np.nan
    
    normalised_stats = {}
    for stat_df in list(statistics_window.keys()):
        stat_dict_df = statistics_window[stat_df]
        zscore_df = pandas_zscore(stat_dict_df)
        normalised_stats[stat_df] = zscore_df
    return statistics_window, normalised_stats


def linear_window_detrending(df, BW ,center = True,
                             wind = None, std_val=None, axis = 0):
    linear_mean = df.rolling(center = center,
                                 window = BW,
                                 win_type=wind,
                             axis = axis, 
                                 min_periods=1
                                ).mean(std = std_val) 
    remove_mean_window = df - linear_mean
    return remove_mean_window
