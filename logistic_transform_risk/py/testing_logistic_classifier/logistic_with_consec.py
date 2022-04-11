from funcs_logs import realisation_detrending,statistics_on_window, read_weights_from_classifier_training
from funcs_logs import logistic_weighted_transform, loop_EWSs_store_results_logistic
from itertools import combinations, groupby
import multiprocessing
from itertools import product
import pandas as pd
import numpy as np
import glob
from operator import itemgetter
import sys
sys.path.append("..")
import helper as h


print('Reading Testing data:')
usr = '../../../data/'
save_results = usr + 'results/logistic/'

npyfiles =glob.glob(usr+'npyFiles/*.npy')
SIS_dataFix = np.load([file for file in npyfiles if 'Fix_SIS_500' in file][0], allow_pickle =True)
SIS_dataExt = np.load([file for file in npyfiles if 'Ext_SIS_500' in file][0], allow_pickle =True)
SIS_dataFixCHANGE = np.load([file for file in npyfiles if 'FixChange_SIS_500' in file][0], allow_pickle =True)

Time = int(sys.argv[1])
print('running for time-series of length:', Time)
realisations = 500
step = int(500/Time)
BT = 300
N =10000
reduced_indicators = (sys.argv[2])
use_incidence = 1
use_normalise = 1

# Bandwidth for moving window calculations
BW = int(0.3*Time)
if BW%2 ==0:
    BW = BW
else:
    BW = BW +1
print('bw:',BW)

# reshape data
Fix_data = np.zeros(shape = (realisations, (Time)))
FixCHANGE_data = np.zeros(shape = (realisations, (Time)))
Ext_data = np.zeros(shape = (realisations, (Time)))
for r in range(realisations):
    Fix_data[r,:] =SIS_dataFix[r][2][BT:][::step]/N
    Ext_data[r,:] =SIS_dataExt[r][2][BT:][::step]/N
    FixCHANGE_data[r,:] = SIS_dataFixCHANGE[r][2][BT:][::step]/N
    

print('Calculate statistics:')

stats_resultsFix, normalised_resultsFix = statistics_on_window(dataframe=pd.DataFrame(data = (Fix_data.T)),
                                                        window_size=int(BW),
                                                        detrend_function=realisation_detrending,
                                                           center = False)

stats_resultsExt, normalised_resultsExt = statistics_on_window(dataframe=pd.DataFrame(data = (Ext_data.T)),
                                    window_size=int(BW),
                                    detrend_function=realisation_detrending,
                                                           center = False)
stats_resultsFixCHANGE, normalised_resultsFixCHANGE = statistics_on_window(dataframe=pd.DataFrame(data = (FixCHANGE_data.T)),
                                    window_size=int(BW),
                                    detrend_function=realisation_detrending,
                                                           center = False)

# run with normalised data
stats_allFix= pd.concat(normalised_resultsFix, axis = 0,names=['stat','Time'])
stats_allExt= pd.concat(normalised_resultsExt, axis = 0,names=['stat','Time'])
stats_allFixCHANGE = pd.concat(normalised_resultsFixCHANGE, axis =0, names = ['stat','Time'])


f_sim = '/storage/'
print('Read data (load the weights and optimal threshold)')
f_root = "../../data"
# Find the weights and optimal thresholds for all composite EWSs
# remove any which had an AUC<reduced_indicators in training
print('removing EWSs with an AUC <', str(reduced_indicators), 'in training')
auc_str = str(reduced_indicators).replace('.','_')
read_weights_from_classifier_training(loc = f_sim, 
                                        f_root = f_root, 
                                        save_results = f_save, 
                                        use_incidence = use_incidence, 
                                        use_normalise = use_normalise,
                                        reduce_by_AUC = reduced_indicators)
thresholds = pd.read_csv((save_results+"thresholds" +
                                    h.incidence_filepath(use_incidence) +
                                    h.normalise_filepath(use_normalise) +
                                    "AUC_" +auc_str+
                                    ".csv"),
                                index_col=0)
weights = pd.read_csv((save_results+"weights" +
                                    h.incidence_filepath(use_incidence) +
                                    h.normalise_filepath(use_normalise) +
                                    "AUC_" +auc_str+
                                    ".csv"), 
                    index_col=0,header=0, names=['ignore','stat','weights'])


weights = weights[[col for col in weights.columns if col !='ignore']]
combinations_results = weights.pivot(columns='stat')
combinations_results.columns = combinations_results.columns.droplevel()

print('Read data (composite stats)')
composite_stats = thresholds.index.values
len(composite_stats)


def main(test_stat, consecutive_length):
    name_result = ''
    for s in test_stat:
        name_result =  (s) + ' + '+name_result
    name_result = name_result[:-3]
    if name_result in composite_stats:

        weights_run = combinations_results.xs(name_result)
        threshold = thresholds.xs(name_result)['log']

        order_of_sim = ['Ext', 'Fix', 'FixCHANGE']
        for index_sim, simulations in enumerate([stats_allExt, stats_allFix, stats_allFixCHANGE]):
            logistic_weighted_EWSs = logistic_weighted_transform(df = simulations,
                                                weights=weights_run,
                                                signals = test_stat
                                                )
            run_logistic_threshold = loop_EWSs_store_results_logistic(df = logistic_weighted_EWSs, 
                                            consecutive_length = consecutive_length,
                                            realisations = realisations, 
                                            time_range = np.arange(0, Time +1,1),
                                            threshold = threshold,
                                            statname = name_result)
            
            for result in ['start', 'end', 'total']:
                logistic_results =run_logistic_threshold[result]
                logistic_results.replace(to_replace=[None], value = np.nan, inplace = True)
                logistic_results.to_csv((save_results+
                                        'consecutive'+
                                        order_of_sim[index_sim]+
                                        'weighted_combination_' +
                                        name_result.replace(' ', '').replace('+','_')+
                                        '_'+
                                        str(consecutive_length) +
                                        't_'+str(Time)+
                                        '_'+
                                        str(result)+
                                        '.csv'))
            


if __name__ == '__main__':
    lengths = [x for x in np.arange(1, 20)]
    num_threads = 12
    f_root = "../../data"
    all_stats_combos= np.load(f_root+'/training_signals.npy', allow_pickle = True)

    with multiprocessing.Pool(processes=num_threads) as pool:
        results = pool.starmap(main,product(all_stats_combos, lengths))
