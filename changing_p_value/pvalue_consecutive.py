import numpy as np
import glob
import pandas as pd
from itertools import product
import scipy.io
import multiprocessing
from itertools import product
from funcs_pvalue import open_pvalue_files, loop_EWSs_store_results_pvalue


def main(length, time):
    Ext_res, Fix_res, Fixchange_res = open_pvalue_files(time)
    stats_of_interest = list(Fix_res.keys())
    
    sim_list = ['EXT', 'FIX', 'FIXCHANGE']
    for index_data, pvalue_data in enumerate([Ext_res, Fix_res, Fixchange_res]):
        run_consec_pvalue = loop_EWSs_store_results_pvalue(dict_input=pvalue_data, 
                                              consecutive_length=length,
                                              all_stats=stats_of_interest,
                                              realisations=realisations,
                                            time_range=np.arange(time//5,
                                                                time+(time//10), 1))
        for result in ['start', 'end', 'total']:
            pvalue_results = pd.concat(run_consec_pvalue[result], axis =0)
            pvalue_results.replace(to_replace=[None], value = np.nan, inplace = True)
            pvalue_results = pvalue_results.groupby(level =0).mean()
            pvalue_results.to_csv(('../data/results/pvalues/'+
                                  sim_list[index_data]+
                                  '_negVM_posIDCVACD_' +
                                  str(length) +
                                  't_'+str(time)+'_'+
                                  str(result)+
                                  '.csv'))  
     

        
if __name__ == '__main__':
    
    times =  [20, 50, 100, 250]
    num_threads = 12
    for t in times:
        lengths =  [x for x in np.arange(1, (t//2)+1)]
        with multiprocessing.Pool(processes=num_threads) as pool:
            results = pool.starmap(main_alltime,product(lengths, [t]))
    
