import pandas as pd
import numpy as np

def bootstrap( time, statistic,
             bootstraps=1000, confidence=0.95,
             pos = 1):
    '''bootstrapping to get confidence interval
    For each prediction of targets, sample (with replacement) from the years. Index the years in the true data (Thomson targets).
    Calculate the AUC for these two timeseries and repeat for bootstaps=1000 times 
    '''
    scores = []
    for n in range(bootstraps):
        indices = np.random.randint(0, len(statistic), len(statistic))
        ktau = statistic.iloc[indices].reset_index(drop=True).corrwith(pd.Series(time), method = 'kendall').values

        scores.append(ktau)
    return scores

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
                                                                     consecutive_length=consecutive_length)
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