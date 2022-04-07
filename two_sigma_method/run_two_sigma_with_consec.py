import numpy as np
import pandas as pd
import glob
from funcs_twosigma import long_run_averages, exceed_threshold_consecutive_years, loop_EWSs_store_results
import sys
sys.path.append("..")
from funcs_all import realisation_detrending,statistics_on_window
from itertools import combinations, groupby
import multiprocessing
from itertools import product

# locate simulation files
usr = '../data/'
simulation_files = glob.glob(usr+'*.npy')
SIS_dataFixChange = np.load([file for file in simulation_files if 'FixChange_SIS_500gamm' in file][0], allow_pickle = True)
SIS_dataFix = np.load([file for file in simulation_files if 'Fix_SIS_500' in file][0], allow_pickle =True)
SIS_dataExt = np.load([file for file in simulation_files if 'Ext_SIS_500' in file][0], allow_pickle =True)

# list EWSs to investigate
stats_of_interest = ['Variance', 'AC(1)', 'CV', 'Kurtosis', 'Decay Time']
all_stats_combos = sum([list(map(list, combinations(stats_of_interest, i))) for i in range(len(stats_of_interest) + 1)], [])
all_stats_combos =all_stats_combos[1:]

# rename all combinations of EWSs
name_stat = []
for index_stat, stat in enumerate(all_stats_combos):
    
    name_result = ''
    for s in stat:
        name_result =  (s[:2]) + ' + '+name_result
    name_stat.append(name_result[:-3])
    
# Model parameters
realisations = 500 #number of realisations
BT = 300 # burn time (simulation)
N = 10000 #population size



    

def main(Time, length):
    print('Time = ', Time)
    print('length = ', length)
    step = int(500/Time)
    BW = int(0.3*Time) # bandwidth to calculate moving window statistics with (30% of entire length of time-series)
    if BW%2 ==0:
        BW = BW
    else:
        BW = BW +1
    print('Bandwidth = ',BW)
    
    order_of_sim = ['Ext', 'Fix', 'FixCHANGE']
    for index_sim, simulations in enumerate([SIS_dataExt, SIS_dataFix, SIS_dataFixChange]):
        # reformat data 
        data = np.zeros(shape = (realisations, (Time)))
        for r in range(realisations):
            data[r,:] = simulations[r][2][BT:][::step]

        # calculate the EWSs
        ews, normalised_ews = statistics_on_window(dataframe = pd.DataFrame(data = (data.T)),
                                                   window_size = int(BW),
                                                    detrend_function = realisation_detrending,
                                                    center = False)


        normalised_df =  pd.concat(normalised_ews, axis = 0)
        
        # run 2-sigma method with consecutive constraint
        run_two_sigma = loop_EWSs_store_results(normalised_df = normalised_df, 
                                               consecutive_length = length,
                                               all_stats_combos = all_stats_combos,
                                               Time = Time, 
                                               name_stat = name_stat)

        # save to csv
        for result in ['start', 'end', 'total']:
            results_two_sigma = pd.concat(run_two_sigma[result], axis =0)
            results_two_sigma.replace(to_replace=[None], value = np.nan, inplace = True)
            results_two_sigma = results_two_sigma.groupby(level =0).mean()
            results_two_sigma.to_csv(('../data/results/2_sigma_results/'+
                                      order_of_sim[index_sim] +
                                      '_' + 
                                    str(Time)+
                                    '_negV_' +
                                    str(length) +
                                    't_'+
                                    str(result)+
                                    '.csv'))



if __name__ == '__main__':
    realisations = 500

    BT = 300
    N = 10000
    # times =  [ 20, 50, 100, 250] # time-series lengths
    times = [20]
    lengths = [1]
    # lengths = [x for x in np.arange(1, 15+1)] # number of consecutive points to loop through
    num_threads = 12 #number of threads to run embarrassingly parallel

    

    with multiprocessing.Pool(processes=num_threads) as pool:
        results = pool.starmap(main, product(times,lengths))
