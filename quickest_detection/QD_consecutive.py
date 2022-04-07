import numpy as np
import glob
import pandas as pd
from itertools import product
import scipy.io
import multiprocessing
from itertools import product
from funcs_qd import loop_EWSs_store_results_QD

files = glob.glob('../data/results/quickest_detection/*')
Ext_files = [file for file in files if 'EXT' in file]
Fix_files = [file for file in files if 'FIX' in file]
realisations = 500


def main( length, Time):
    print('Time = ', Time)
    print('length = ', length)
    A_threshold = np.arange(1.5,10, 0.5)
    step = int(500/Time)
    
    name_matlab = ['ext', 'fix', 'fixCHANGE']
    data_files = [Ext_files,Fix_files, Fix_files ]
    for index_, files_open in enumerate(data_files):
        # open matlab .mat file
        mat = scipy.io.loadmat([file for file in files_open if name_matlab[index_].upper()+str(Time) in file][0])
        data = pd.DataFrame(data = mat['log_RR_' +name_matlab[index_]], index = range(realisations),
                            columns=range(Time))


        # run QD on the loglikelihood and run consecutive point constraint
        run_consec_QD = loop_EWSs_store_results_QD(df=data, 
                                              consecutive_length=length,
                                              realisations=realisations,
                                           time_range=np.arange(0, Time+1, 1),
                                            A =A_threshold )
        # save the data
        QD_results = pd.concat(run_consec_QD['start'], axis =0)
        QD_results.replace(to_replace=[None], value = np.nan, inplace = True)
        QD_results = QD_results.groupby(level =0).mean()
        QD_results.to_csv(('../data/results/quickest_detection/'
                                    'consecutive'+
                                    name_matlab[index_].upper()+
                                    '_'+
                                    str(length) +
                                    't_'+str(Time)+'_'+
                                    str('start')+
                                    '.csv'))  

    

if __name__ == '__main__':
    
    times =  [20, 50, 100, 250]
    num_threads = 12
    for t in times:
        lengths =  [x for x in np.arange(1, ((t//5))+1)]
        with multiprocessing.Pool(processes=num_threads) as pool:
            results = pool.starmap(main,product(lengths, [t]))
    
