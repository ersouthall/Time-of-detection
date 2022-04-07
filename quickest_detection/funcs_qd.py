import numpy as np
from operator import itemgetter
import pandas as pd

def exceed_threshold_consecutive_years(stat_timeseries, years, consecutive_length = 2, threshold=8):
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
    ENDresults = {}
    TOTALresults = {}
    STARTresults = {}

    for colname in (range(realisations)):
        start_year = []
        end_year = []
        total_year = [] 
        for aa in A:

            try:

                pval_sim = df.values[colname,:]
#                 print(aa)
                start_res, finish_res, total_res = exceed_threshold_consecutive_years(stat_timeseries = pval_sim, 
                                                                            years = time_range,
                                                                     consecutive_length=consecutive_length,
                                                                           threshold=aa)
                start_year.append(start_res)
                end_year.append(finish_res)
                total_year.append(total_res)
            except:
                start_year.append(np.nan)
                end_year.append(np.nan)
                total_year.append(np.nan)
        STARTresults[colname] = pd.DataFrame(np.reshape(start_year, 
                                                                  (1,len(A))),
                                                       columns=A,
                                                       index = [str(colname)])
        ENDresults[colname] = pd.DataFrame(np.reshape(end_year, 
                                                                  (1,len(A))),
                                                       columns=A,
                                                       index = [str(colname)])
        TOTALresults[colname] = pd.DataFrame(np.reshape(total_year, 
                                                                  (1,len(A))),
                                                       columns=A,
                                                       index = [str(colname)])
    return {'start': STARTresults, 'end': ENDresults, 'total': TOTALresults}