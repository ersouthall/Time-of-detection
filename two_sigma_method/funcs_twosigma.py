import numpy as np
from itertools import combinations, groupby
from operator import itemgetter
import pandas as pd

def long_run_averages(standardised):
    n = len(standardised)
    running_mean = np.zeros(n)
    running_std= np.zeros(n)
    for j in range(n):
        running_mean[j] = np.nanmean(standardised[1:j+1])
        running_std[j] = np.nanstd(standardised[1:j+1])

        
    return running_std, running_mean

def exceed_threshold_consecutive_years(stat_timeseries, years, consecutive_length = 2):
    
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
                start_res, finish_res, total_res = exceed_threshold_consecutive_years(stat_sim, range(Time),
                                                                     consecutive_length=consecutive_length)
                composite_start_year.append(start_res)
                composite_end_year.append(finish_res)
                composite_total_year.append(total_res)
            except:
                composite_start_year.append(np.nan)
                composite_end_year.append(np.nan)
                composite_total_year.append(np.nan)

        composite_STARTresults[colname] = pd.DataFrame(np.reshape(composite_start_year, 
                                                                  (1,31)),
                                                       columns=name_stat,
                                                       index = [colname])
        composite_ENDresults[colname] = pd.DataFrame(np.reshape(composite_end_year, 
                                                                  (1,31)),
                                                       columns=name_stat,
                                                       index = [colname])
        composite_TOTALresults[colname] = pd.DataFrame(np.reshape(composite_total_year, 
                                                                  (1,31)),
                                                       columns=name_stat,
                                                       index = [colname])
        
    return {'start': composite_STARTresults, 'end': composite_ENDresults, 'total': composite_TOTALresults}