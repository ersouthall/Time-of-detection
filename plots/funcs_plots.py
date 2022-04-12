import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from itertools import combinations, groupby
import seaborn as sns
from operator import itemgetter
from sklearn.metrics import auc
import sys
sys.path.append("..")
from funcs_all import statistics_on_window, realisation_detrending
realisations = 500
BT = 300
N = 10000

def logistic_weighted_transform(df, weights, signals, Time, standardise=False):
    
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
    

def lr_emergence_risk(df):
    return 1 / (1 + np.exp(-df))

def logistic_plot(test_stat, best_number_consec):
    name_result = ''
    for s in test_stat:
        name_result =  (s) + ' + '+name_result
    name_result = name_result[:-3]
    name_result
    
    usr = "../data/"
    results_dir = usr+"results/logistic/"
    weights = pd.read_csv((results_dir+
                       'weights' +
                       '_incidence' + 
                       '_norm'+
                       'AUC_0_6' +
                       '.csv'), 
                      index_col=0,header=0, names=['ignore','stat','weights'])

    weights = weights[[col for col in weights.columns if col !='ignore']]
    
    combinations_results = weights.pivot(columns='stat')
    combinations_results.columns = combinations_results.columns.droplevel() 
    weights_run = combinations_results.xs(name_result)

    thresholds = pd.read_csv((results_dir+
                       'thresholds' +
                       '_incidence' + 
                       '_norm'+
                       'AUC_0_6' +
                       '.csv'), index_col=0)

    threshold = thresholds.xs(name_result)['log']
    threshold = np.maximum(threshold, 0.5)
    logistic_Fix = logistic_weighted_transform(df = stats_allFix,
                                           weights=weights_run,
                                           signals = test_stat
                                           )
    logistic_FixCHANGE = logistic_weighted_transform(df = stats_allFixCHANGE,
                                               weights=weights_run,
                                               signals = test_stat
                                               )
    logistic_Ext= logistic_weighted_transform(df = stats_allExt,
                                              weights=weights_run,
                                             signals = test_stat
                                             )

    ext_return = consecutive_results_logistic(df = logistic_Ext, 
                     consecutive_length = best_number_consec,
                     realisations = 500, 
                     time_range = np.arange(0, Time+1,1),
                     threshold = threshold,
                     statname = name_result)

    fix_return = consecutive_results_logistic(df = logistic_Fix, 
                         consecutive_length = best_number_consec,
                         realisations = 500, 
                         time_range = np.arange(0, Time+1,1),
                         threshold = threshold,
                         statname = name_result)


    fixCHANGE_return = consecutive_results_logistic(df = logistic_FixCHANGE, 
                         consecutive_length = best_number_consec,
                         realisations = 500, 
                         time_range = np.arange(0, Time+1,1),
                         threshold = threshold,
                         statname = name_result)
    
    ext_dates  =ext_return['start']
    ext_dates = ext_dates.reset_index()
    ext_dates['true positives']  = ext_dates[name_result]


    fix_dates  = fix_return['start']
    fix_dates = fix_dates.reset_index()
    fix_dates['false positives 1']  = fix_dates[name_result]

    fixCHANGE_dates  =fixCHANGE_return['start']
    fixCHANGE_dates = fixCHANGE_dates.reset_index()
    fixCHANGE_dates['false positives 2']  = fixCHANGE_dates[name_result]
    
    
    both = pd.concat([ext_dates, fix_dates, fixCHANGE_dates], axis = 1)
    both = both.loc[:,~both.columns.duplicated()]
    colnames = ['true positives','false positives 1','false positives 2']
    plot_df = pd.melt(both[colnames])
    plot_df['datatype'] = plot_df['variable']
    plot_df['statistic'] = name_conversion.loc[name_result].values[0]
    return plot_df

def power_metric(Ext, Fix):
    TPR = Ext/500
    FPR = Fix/500
    return (TPR-FPR)

def exceed_threshold_consecutive_years_log(stat_timeseries, years, consecutive_length = 2):
    
    std_val, mean_val = long_run_averages(stat_timeseries)
    # Get indices where stat > mean + 2*sigma and indices where stat < mean - 2*sigma
    indices_greater = [ n for n,i in enumerate(range(len(stat_timeseries))) if (stat_timeseries[i])>(mean_val+2*std_val)[i] ]
    indices_smaller =  [ n for n,i in enumerate(range(len(stat_timeseries))) if (stat_timeseries[i])<(mean_val-2*std_val)[i] ]
    all_indices = np.setdiff1d(np.unique(indices_greater + indices_smaller ),[1])    
    try:
        if consecutive_length ==1:
            start = all_indices[0]
            finish = all_indices[-1]
            total = len(all_indices)
            return start, finish, total
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
                return start, finish, total
        
    except:
        return np.nan

def long_run_averages(standardised):
    n = len(standardised)
    running_mean = np.zeros(n)
    running_std= np.zeros(n)
    for j in range(n):
        running_mean[j] = np.nanmean(standardised[1:j+1])
        running_std[j] = np.nanstd(standardised[1:j+1])

        
    return running_std, running_mean
    
def loop_EWSs_store_results(normalised_df, consecutive_length,all_stats_combos,Time, name_stat ):
    composite_ENDresults = {}
    composite_TOTALresults = {}
    composite_STARTresults = {}
    
    for colname in (normalised_df.columns):
        composite_start_year = []
        composite_end_year = []
        composite_total_year = []
        for index_composite, composite in enumerate(all_stats_combos):
            composite_test = composite.copy()

            try:
                # take the negation of variance (as predicted to decrease prior to disease elim.)
                if 'Variance' in composite_test: 
                    composite_test.remove('Variance')
                    if len(composite_test)>0:
                        stat_sim = normalised_df[colname][composite_test].groupby(level = 1).sum(min_count = 1).values
                        stat_sim = stat_sim - normalised_df[colname]['Variance'].values
                    else:
                        stat_sim =- normalised_df[colname]['Variance'].values
                else:
                    stat_sim = normalised_df[colname][composite_test].groupby(level = 1).sum(min_count = 1).values
                start_res, finish_res, total_res = exceed_threshold_consecutive_years_log(stat_sim, range(Time),
                                                                     consecutive_length=consecutive_length[index_composite])
                composite_start_year.append(start_res)
                composite_end_year.append(finish_res)
                composite_total_year.append(total_res)
            except:
                composite_start_year.append(np.nan)
                composite_end_year.append(np.nan)
                composite_total_year.append(np.nan)

        composite_STARTresults[colname] = pd.DataFrame(np.reshape(composite_start_year, 
                                                                  (1,len(all_stats_combos))),
                                                       columns=name_stat,
                                                       index = [colname])
        composite_ENDresults[colname] = pd.DataFrame(np.reshape(composite_end_year, 
                                                                  (1,len(all_stats_combos))),
                                                       columns=name_stat,
                                                       index = [colname])
        composite_TOTALresults[colname] = pd.DataFrame(np.reshape(composite_total_year, 
                                                                  (1,len(all_stats_combos))),
                                                       columns=name_stat,
                                                       index = [colname])
        
    return {'start': composite_STARTresults, 'end': composite_ENDresults, 'total': composite_TOTALresults}

def twosigma(standardised):
    n = len(standardised)
    running_average = np.zeros(n)
    running_std= np.zeros(n)
    for j in range(n):
        running_average[j] = np.nanmean(standardised[1:j+1])
        running_std[j] = np.nanstd(standardised[1:j+1])

        
    return running_std, running_average

def loop_EWSs_store_results_pvalue(dict_input, consecutive_length,all_stats,realisations, time_range ):
    ENDresults = {}
    TOTALresults = {}
    STARTresults = {}
    
    for colname in (range(realisations)):
        start_year = []
        end_year = []
        total_year = []
        for index_stat, stat in enumerate(all_stats):

            try:

                pval_sim = dict_input[stat].values[colname,:]
                start_res, finish_res, total_res = significance_threshold_consecutive_years(stat_timeseries = pval_sim, 
                                                                            years = time_range,
                                                                     consecutive_length=consecutive_length[index_stat])
                start_year.append(start_res)
                end_year.append(finish_res)
                total_year.append(total_res)
            except:
                start_year.append(np.nan)
                end_year.append(np.nan)
                total_year.append(np.nan)
        STARTresults[colname] = pd.DataFrame(np.reshape(start_year, 
                                                                  (1,len(all_stats))),
                                                       columns=all_stats,
                                                       index = [str(colname)])
        ENDresults[colname] = pd.DataFrame(np.reshape(end_year, 
                                                                  (1,len(all_stats))),
                                                       columns=all_stats,
                                                       index = [str(colname)])
        TOTALresults[colname] = pd.DataFrame(np.reshape(total_year, 
                                                                  (1,len(all_stats))),
                                                       columns=all_stats,
                                                       index = [str(colname)])
        
    return {'start': STARTresults, 'end': ENDresults, 'total': TOTALresults}

def significance_threshold_consecutive_years(stat_timeseries, years, consecutive_length = 2, threshold=0.05):
    # Get indices where p-value crosses the significance threshold 

    all_indices = [ n for n,i in enumerate(range(len(stat_timeseries))) if (stat_timeseries[i])<=(threshold) ]
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
    
def open_pvalue_files(Time):
    Extcsv_files = glob.glob("../data/results/pvalues/Ext_"+str(Time)+"*.csv")
    Fixcsv_files = glob.glob("../data/results/pvalues/Fix_"+str(Time)+"*.csv")
    FixChangecsv_files = glob.glob('../data/results/pvalues/FixChange_'+str(Time)+'*.csv')
    Ext_res = {}
    for name in Extcsv_files:
        name_stat = (name.split('/')[-1].split('_')[-3])
        open_file = pd.read_csv(name, names=[x for x in np.arange(Time//5,Time,1)])
        Ext_res[name_stat] = open_file
    Fix_res = {}
    for name in Fixcsv_files:
        name_stat = (name.split('/')[-1].split('_')[-3])
        open_file = pd.read_csv(name, names=[x for x in np.arange(Time//5,Time,1)])
        Fix_res[name_stat] = open_file
        
    FixChange_res = {}
    for name in FixChangecsv_files:
        name_stat = (name.split('/')[-1].split('_')[-3])
        open_file = pd.read_csv(name, names=[x for x in np.arange(Time//5,Time,1)])
        FixChange_res[name_stat] = open_file
    return Ext_res, Fix_res,FixChange_res


def exceed_threshold_consecutive_years_QD(stat_timeseries, years, consecutive_length = 2, threshold=8):
    # Get indices where LL ratio exceeds the threshold 

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
    
def loop_EWSs_store_results_QD(df, consecutive_length,realisations, time_range, A):
    start_year = []
    end_year = []
    total_year = [] 

    for colname in (range(realisations)):

        try:
            qd_sim = df.values[colname,:]
            start_res, finish_res, total_res = exceed_threshold_consecutive_years_QD(stat_timeseries = qd_sim, 
                                                                        years = time_range,
                                                                 consecutive_length=consecutive_length,
                                                                       threshold=float(A))
            start_year.append(start_res)
            end_year.append(finish_res)
            total_year.append(total_res)
        except:
            start_year.append(np.nan)
            end_year.append(np.nan)
            total_year.append(np.nan)
        
    return {'start': start_year, 'end': end_year, 'total': total_year}