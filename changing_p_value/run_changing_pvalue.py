import numpy as np
import pandas as pd
import glob
from funcs_all import realisation_detrending,statistics_on_window
from funcs_pvalue import bootstrap
import multiprocessing
from itertools import product
import sys 

#  Input time-series length to be considered on the command line
Time = int(sys.argv[1])

# Open simulations npy files
npyfiles = glob.glob('../data/*.npy')
SIS_dataFixChange = np.load([file for file in npyfiles if 'FixChange_SIS_500' in file][0], allow_pickle =True)
SIS_dataExt = np.load([file for file in npyfiles if 'Ext_SIS_500' in file][0], allow_pickle =True)
SIS_dataFix = np.load([file for file in npyfiles if 'Fix_SIS_500' in file][0], allow_pickle =True)

# parameters
stats_of_interest = ['Variance', 'AC(1)', 'CV', 'Index of dispersion', 'Decay Time', 'Mean']
realisations = 500
BT = 300
N = 10000
step = int(500/Time)
study_time = range(Time)

# bandwidth to calculate the EWSs (moving average window)
BW = int(0.3*Time)
if BW%2 ==0:
    BW = BW
else:
    BW = BW +1

print('bw:',BW)

# Reformat simulation data
Fix_data = np.zeros(shape = (realisations, (Time)))
FixChange_data = np.zeros(shape = (realisations, (Time)))
Ext_data = np.zeros(shape = (realisations, (Time)))

for r in range(realisations):
    Fix_data[r,:] =SIS_dataFix[r][2][BT:][::step]
    Ext_data[r,:] =SIS_dataExt[r][2][BT:][::step]
    FixChange_data[r,:] = SIS_dataFixChange[r][2][BT:][::step]

# Calculate EWSs using realisation detrending
stats_resultsFix, normalised_resultsFix = statistics_on_window(dataframe=pd.DataFrame(data = (Fix_data.T)),
                                                        window_size=int(BW),
                                                        detrend_function=realisation_detrending,
                                                           center = False)

stats_resultsFixChange, normalised_resultsFixChange = statistics_on_window(dataframe=pd.DataFrame(data = (FixChange_data.T)),
                                                        window_size=int(BW),
                                                        detrend_function=realisation_detrending,
                                                           center = False)

stats_resultsExt, normalised_resultsExt = statistics_on_window(dataframe=pd.DataFrame(data = (Ext_data.T)),
                                    window_size=int(BW),
                                    detrend_function=realisation_detrending,
                                                           center = False)



def main(stat,sim):
    # Select correct dataframe
    if sim =='Ext':
        stat_df = stats_resultsExt
    elif sim=='Fix':
        stat_df = stats_resultsFix
    else:
        stat_df = stats_resultsFixChange
    # select the statistic to study
    df = stat_df[stat]

    # matrix of p-values through time (realisations, time) 
    # require at least 20% of points to calculate the p-value (hence floor the Time//5)
    p_values = np.zeros((500, len(range(Time//5, Time, 1))) )

    for index_time, time in enumerate(range(Time//5, Time, 1)):
        # Calculate Kendall's tau score up to each time point
        ktau_all_sim = df.iloc[:time].corrwith(pd.Series(study_time[:time]), method = 'kendall').values
        # Run bootstrap on the time-series up to each time point, and recalculate kendall's tau
        S = bootstrap(time = study_time[:time], statistic=df.iloc[:time])
        S = np.array(S)
        if stat in ['Variance', 'Mean']: #if the statistic is expected to decrease, then take left tail
            return_val = (np.sum(S <= ktau_all_sim,0)/1000)
        elif stat in ['Index of dispersion', 'CV', 'AC(1)','Decay Time' ]: #if the statistic is expected to decrease, then take right tail
            return_val = (np.sum(S >= ktau_all_sim,0)/1000)
        else:
            print('EWS not in list. Please check!')
        p_values[:, index_time] = return_val
    
    # Save output
    np.savetxt("../data/results/pvalues/"+sim+"_"+str(Time)+"_"+stat[:2]+".csv", p_values, delimiter=",")
        


if __name__ == '__main__':
    realisations = 500

    BT = 300
    N = 10000
    num_threads = 12
    stats_of_interest = ['Variance', 'AC(1)', 'CV', 'Index of dispersion', 'Decay Time', 'Mean']
    sim_study = ['Ext', 'Fix', 'FixChange']


    with multiprocessing.Pool(processes=num_threads) as pool:
        results = pool.starmap(main, product(stats_of_interest,sim_study))

