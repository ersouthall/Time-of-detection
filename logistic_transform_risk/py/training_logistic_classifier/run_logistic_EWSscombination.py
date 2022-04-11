import numpy as np
import sys 
import classifier_training as ct
import ews_logistic_regression as elr
import sys
sys.path.append("..")
import helper as h
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from itertools import combinations, groupby

choice = int(sys.argv[1]) # which training signal 

f_root = usr+"../../data"

def create_list_all_EWSs_combinations(list_signals):
    all_stats_combos = sum([list(map(list, combinations(list_signals, i))) for i in range(len(list_signals) + 1)], [])
    all_stats_combos =all_stats_combos[1:]
    np.save(f_root+'/training_signals.npy', all_stats_combos)
# Either open or create npy file containing all combinations of EWSs
try:
    training_data= np.load(f_root+'/training_signals.npy', allow_pickle = True)
except:
    create_list_all_EWSs_combinations(list_signals = h.signals())
    training_data = np.load(f_root+'/training_signals.npy', allow_pickle = True)

f_sim = '/storage/'
params_sim = h.simulator_args()
c_min = 0.0001

# select which combination of EWSs to consider
signals = training_data[choice]
print('Signals testing', signals)

use_incidence = 1
use_normalise =1 

if use_normalise:
    edf = pd.read_csv(f_root + "/zscore_ews_data" +
                          h.incidence_filepath(use_incidence) + ".csv", index_col=0)
else:
    edf = pd.read_csv(f_root + "/ews_data" +
                    h.incidence_filepath(use_incidence) + ".csv", index_col=0)
            

# Get data points used for training (last 10 years)
# ews_data = edf[edf.Time.between(10 * 365, 20*365)].copy()
ews_data = edf[edf.Time.between((params_sim["BurnTime"]),
                                (params_sim["BurnTime"]+params_sim["Time"]))].copy()
# Drop NAs
ews_data = ews_data[~ews_data.isin([np.inf, -np.inf, np.nan]).any(axis=1)]
ews_data = ews_data[~ews_data[signals].isna().any(axis=1)]


coefs, lr_clf = elr.ews_logistic_regression(ews_data[ews_data["Time"] > 
                                                        (params_sim['BurnTime'])],
                                            standardise=True,
                                            signals=signals, do_pca=False,
                                            penalty="l1", C=c_min,
                                            solver="liblinear")
c2 = pd.Series(coefs)
print('Print coeficients', coefs)
ews_data["all"] = ews_data[signals].dot(c2[signals]) \
                    + c2["intercept"]
                    
                    
fpr_dict = {}
tpr_dict = {}
thresh_dict = {}
roc_auc_dict = {}
df_c_dict = {}
for signal in signals + [ "all"]:
    y_score = ews_data[signal].values
    y_true = ews_data["is_test"]
    # get threshold for ROC curve 
    fpr_dict[signal], tpr_dict[signal], thresh_dict[signal] = \
        metrics.roc_curve(y_true, y_score, pos_label=1)
    roc_auc_dict[signal] = roc_auc_score(y_true, y_score)
    # get threshold as the min(fpr - tpr)
    df_c_dict[signal] = thresh_dict[signal][np.argmin(fpr_dict[signal]
                                                        - tpr_dict[signal])]

# Write df_c_dict, coefs and auc_dict to files:
pd.Series(df_c_dict).to_csv(f_sim + "optimum_thresholds" +
                            h.incidence_filepath(use_incidence)+
                             h.normalise_filepath(use_normalise) + '_'+str(choice) + ".csv")

pd.Series(coefs).to_csv(f_sim + "ews_weights" +
                        h.incidence_filepath(use_incidence)+
                         h.normalise_filepath(use_normalise) + '_'+str(choice) + ".csv")

pd.DataFrame(roc_auc_dict, index = [0]).to_csv(f_sim + "roc_auc" +
                                                h.normalise_filepath(use_normalise) + 
                                  h.incidence_filepath(use_incidence)+'_'+str(choice) + ".csv")
