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