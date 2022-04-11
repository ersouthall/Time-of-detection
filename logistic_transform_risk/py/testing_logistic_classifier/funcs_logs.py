import numpy as np
import pandas as pd 
from scipy.io import loadmat
from scipy.stats import skew, kurtosis, kendalltau
from itertools import combinations, groupby
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter as gf
import sys
sys.path.append("..")
import helper as h

def read_weights_from_classifier_training(loc, 
                                          f_root, 
                                          f_save, 
                                          use_incidence, 
                                          use_normalise, reduce_by_AUC = 0):
    '''
    Combines multiple pandas dataframe produced during the logistic regression on the training data, ...
    ... which contain the AUC, weights and optimal thresholds for each EWS composite. 
    Reduces the total number of composite EWSs by removing any which have an AUC less than reduce_by_AUC 
    INPUT
    :param loc: str, location of output files from the logistic regression (see run_logistic_EWSscombination.py)
    :param f_root: str, location of all EWSs composites considered
    :param use_incidence: boolean, filter files based on whether incidence data was used or not in training 
    :param use_normalise: boolean, filter files based on whether data was normalised or not in training
    :param reduce_by_AUC: float, default 0. Value of AUC to threshold files by. Score in [0, 1] 
    RETURN
    saves single pandas dataframe for all the results in f_root:
    f_root+"/thresholds" + h.incidence_filepath(use_incidence) + h.normalise_filepath(use_normalise) ".csv"
    f_root+"/aucs" + h.incidence_filepath(use_incidence) + h.normalise_filepath(use_normalise) ".csv"
    f_root+"/weights" + h.incidence_filepath(use_incidence) + h.normalise_filepath(use_normalise) ".csv"

    '''
    # find all files for each combination of EWSs with training data
    allfiles = glob.glob(loc + '*.csv')
    allfiles_incidence_norm = [files for files in allfiles if  (h.incidence_filepath(use_incidence)+h.normalise_filepath(use_normalise)) in files]
    weight_files = [files for files in allfiles_incidence_norm if 'weight' in files]
    threshold_files = [files for files in allfiles_incidence_norm if 'optimum' in files]
    auc_files = [files for files in allfiles_incidence_norm if 'auc' in files]
    training_signals= np.load(f_root+'/training_signals.npy', allow_pickle = True)

    auc_scores = {}
    weights = {}
    thresholds = {}
    for file in auc_files:
        choice = int(file.split('_')[-1].split('.')[0]) # select which composite of EWSs
        signal_choice = training_signals[choice]
        name_result = ''
        for s in signal_choice:
            name_result =  (s) + ' + '+name_result
        name_result = name_result[:-3]
        
        weightfile = [file for file in weight_files if '_'+str(choice)+'.' in file][0]
        weights_results = pd.read_csv(weightfile, header=None)
        thresholdfile = [file for file in threshold_files if '_'+str(choice)+'.' in file][0]
        if ~(weights_results[1]==0).any(0):
            weights[name_result] = weights_results

            threshold_df =  pd.read_csv(thresholdfile, header = None)
            threshold_df['log'] = 1/(1+np.exp(-threshold_df[1])) #logistic transform
            thresholds[name_result] = threshold_df
            auc_scores[name_result] = pd.read_csv(file)
            
    dfAUC = pd.concat(auc_scores)
    dfWEIGHTS = pd.concat(weights)
    dfTHRESHOLD = pd.concat(thresholds)
    dfAUC = dfAUC.loc[:, ~dfAUC.columns.str.contains('^Unnamed')] 
    dfAUC['max_choice']=dfAUC.max(1)
    dfAUC['max_type'] = dfAUC.idxmax(1)

    #remove any repeated entries (when the weight assigned is zero)
    dfAUC = dfAUC[dfAUC['max_type']=='all']
    reduce_combos = dfAUC[(dfAUC['max_choice']>reduce_by_AUC)].sort_values(['all'], ascending = False)

    weights_reduce = {}
    for EWS in reduce_combos.index.get_level_values(0).values:
        weights_reduce[EWS] = dfWEIGHTS.xs(EWS)
        
    auc_str = str(reduce_by_AUC).replace('.','_')

    pd.concat(weights_reduce).to_csv((f_save + 
                                      "/weights" +
                                      h.incidence_filepath(use_incidence) +
                                      h.normalise_filepath(use_normalise) +
                                      "AUC_" +auc_str+
                                      ".csv"))
    dfTHRESHOLD[dfTHRESHOLD[0]=='all'].droplevel(1).loc[reduce_combos.index.get_level_values(0).values].to_csv((f_save + 
                                      "/thresholds" +
                                      h.incidence_filepath(use_incidence) +
                                      h.normalise_filepath(use_normalise) +
                                      "AUC_" +auc_str+
                                      ".csv"))
    

def autocorrelation_tau(detrended_data, param_dict):
    '''
    param_dict requires 
        repeats (or number of places)
        T (total time period)
        BT (set to zero)
        windowSize (for the calculating )
        lag_tau '''
    AC_results = np.nan*np.zeros((param_dict['repeats'], param_dict['T']+param_dict['BT']))
    Acov_results = np.nan*np.zeros((param_dict['repeats'], param_dict['T']+param_dict['BT']))
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
        AC_results[:,i] = xtest_autocorr
        Acov_results[:,i] = numerator
    return Acov_results, AC_results 

def logistic_weighted_transform(df, weights, signals, standardise=False):
    
    stat_name_to_name = {'Variance': 'variance',
                    'Mean': 'mean',
                    'Index of dispersion': 'index_of_dispersion',
                    'CV': 'coefficient_of_variation',
                    'Kurtosis': 'kurtosis',
                    'Skewness': 'skewness',
                    'AC(1)': 'autocorrelation',
                    'AC(2)': 'ac2',
                    'Acov1': 'autocovariance',
                    'Decay Time': 'decay_time',
                    'standard_deviation': 'standard_deviation'}

    new_data = {}
    for stat in stat_name_to_name.keys():
        test = df.reset_index()
        test_melt = pd.melt(test[test.stat==stat][np.arange(0, realisations)],
                                var_name ='run',
                                value_name=stat_name_to_name[stat])
        test_melt['Time'] = np.tile(np.arange(Time),realisations)
        new_data[stat_name_to_name[stat]] = test_melt

    concat_data = pd.concat(new_data, axis =1, )
    concat_data.columns = concat_data.columns.droplevel()
    concat_data = concat_data.loc[:,~concat_data.columns.duplicated()]
    concat_data['standard_deviation'] = np.sqrt(concat_data['variance'])

    if standardise:
        x = concat_data[signals]
        scaler = StandardScaler()
        scaler.fit(x)
        concat_data[signals]= scaler.transform(x)
    concat_data['all'] = (concat_data[signals]*weights[signals].values).sum(axis = 1)+ weights['intercept']
    concat_data.loc[concat_data['Time'].isin(range(int(3))),'all'] = np.nan
    concat_data['logistic'] = lr_emergence_risk(concat_data['all'])
    return concat_data


def lr_emergence_risk(df):
    return 1 / (1 + np.exp(-df))

def loop_EWSs_store_results_logistic(df, consecutive_length,realisations, time_range, threshold, statname ):
    start_year = []
    end_year = []
    total_year = [] 
    for colname in (range(realisations)):

        try:

            log_sim = df[df.run == colname]['logistic'].values
            start_res, finish_res, total_res = exceed_threshold_consecutive_years(stat_timeseries = log_sim, 
                                                                        years = time_range,
                                                                 consecutive_length=consecutive_length,
                                                                       threshold = threshold)
            start_year.append(start_res)
            end_year.append(finish_res)
            total_year.append(total_res)
        except:
            start_year.append(np.nan)
            end_year.append(np.nan)
            total_year.append(np.nan)
            
            
            
    STARTresults = pd.DataFrame(start_year,columns=[statname],
                                                   index = range(realisations))
    ENDresults = pd.DataFrame(end_year, columns=[statname],
                                                   index =range(realisations))
    TOTALresults = pd.DataFrame(total_year, columns=[statname],
                                                   index = range(realisations))
        
    return {'start': STARTresults, 'end': ENDresults, 'total': TOTALresults}

def exceed_threshold_consecutive_years(stat_timeseries, years, consecutive_length = 2, threshold=0.5):
    all_indices = [ n for n,i in enumerate(range(len(stat_timeseries))) if (stat_timeseries[i])>=(threshold) ]
    try:
        if consecutive_length ==1:
            start = all_indices[0]
            finish = all_indices[-1]
            total = len(all_indices)
            return years[start], years[finish], years[total]
        else:
            smallest_diff = np.min(np.diff(all_indices))

            if smallest_diff <2:
                index = []
                for k, g in groupby(enumerate(all_indices), lambda x:x[0]-x[1]):
                    group = list(map(itemgetter(1), g))
                    if len(group)>=consecutive_length:
                        index = index+(group)
                start = index[0]
                finish = index[-1]
                total = len(index)
                return years[start], years[finish], total
        
    except:
        return np.nan
    
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
    statistics_window['standard_deviation'] = detrended_data.rolling(center = center,
                             window = window_size,
                             win_type=None,
                             min_periods=1
                            ).std()

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
                                 window = np.maximum(window_size,4),
                                 win_type=None,
                                 min_periods=1
                                ).kurt()
    statistics_window['Skewness'] = detrended_data.rolling(center = center,
                                 window =np.maximum( window_size,3),
                                 win_type=None,
                                 min_periods=1
                                ).skew()


    ### autocorrelation 
    matrix_time_place = detrended_data.values
    matrix_place_time = matrix_time_place.T
    ac_tau_1cov, ac_tau_1corr = autocorrelation_tau(matrix_place_time, param_dict={'repeats': len(dataframe.columns),
                                                             'T': len(dataframe.index),
                                                             'BT': 0,
                                                             'windowSize': window_size,
                                                             'lag_tau': 1})

    statistics_window['AC(1)'] =pd.DataFrame(ac_tau_1corr.T, index=dataframe.index, columns= dataframe.columns)
    statistics_window['Acov1'] =pd.DataFrame(ac_tau_1cov.T, index=dataframe.index, columns= dataframe.columns)
    
    ac_tau_2cov, ac_tau_2corr = autocorrelation_tau(matrix_place_time, param_dict={'repeats': len(dataframe.columns),
                                                             'T': len(dataframe.index),
                                                             'BT': 0,
                                                             'windowSize': window_size,
                                                             'lag_tau': 2})

    statistics_window['AC(2)'] =pd.DataFrame(ac_tau_2corr.T, index=dataframe.index, columns= dataframe.columns)
    
    statistics_window['Decay Time'] = pd.DataFrame(decay_time(ac_tau_1corr).T,
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
