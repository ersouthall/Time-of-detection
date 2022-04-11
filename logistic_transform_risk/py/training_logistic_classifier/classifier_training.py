import ews_logistic_regression as elr
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn import metrics
import sys
sys.path.append("..")
import helper as h



def run_classifier_training(use_incidence, use_cross_val, calculate_ews,
                            normalise_ews, use_normalise, folder,
                            folder_simulation_output, 
                            save_df=False, aggregation_period=1,
                            training_signals=None, hyperparameters=(156,0.0001),
                            mv_method = 'exp',data_detrend = False, center = False):
    """
    Trains classifier to get weightings for ews using all data from last 10 years of simulation

    :param use_incidence: boolean, if true then uses reported_cases/N_i, else uses reported_cases
    :param use_cross_val: boolean, if true reads saved output file from CV to get best ...
    ... window size and regulariser. Else uses the default hyperparameters. 
    :param calculate_ews: boolean, if true calculates the EWS using CV results for windowsize ...
    ... or using hyperparameters.
    :param normalise_ews: boolean, if true it normalises the output EWSs using the long run mean ...
    ... and long run standard deviation  
    :param use_normalise: boolean, if true it uses the normalised save data for the logistic regression
    :param folder: str, location of data
    :param save_df: boolean, default False
    :param aggregation_period: int, default 4, time aggegragation parameter to group the data ...
    ... e.g. 4 groups data into 4 weekly periods
    :param training_signals: array of strings, default is None and uses helper.signals()
    :param hyperparameters: tuple, default (156, 0.0001)
    :param mv_method: str, weighting for detrending (options: 'exp', 'uniform' or 'gaussian')
    :param data_detrend: boolean, default False, if true it indicates that the data has already been detrended
    :param center: boolean, defualt False, if true then EWSs are calculated on a centered moving window, else on a right-window
    :return:
    saves output of EWS calculated with hyperparameters/Cross_validation results for best window...
    ... size. 
    saves classifier coefficients and yintercept to folder + "/ews_weights" +
                            h.incidence_filepath(use_incidence) + ".csv"
    saves optimimum threshold from ROC curve (to achieve highest TPR and lowest FPR) to older + "/optimum_thresholds" +
                                h.incidence_filepath(use_incidence) + ".csv"
    saves AUC score for each EWS, raw timeseries and weighted sum to folder + "/auc_time_series" +
                                  h.incidence_filepath(use_incidence) + ".csv"
    """
    def realisation_detrending(df):
        mean_data  = df.groupby([ 'is_test','Time']).mean()
        grouped_data =  df.groupby([ 'is_test', 'model','Time']).mean()
        detrended_data = {}
        for is_test_value in [0,1]:
            df_is_test = df[df.is_test==is_test_value]
            model_numbers = df_is_test.model.unique()

            mean_data_is_test = pd.concat([mean_data.xs(is_test_value, level = 'is_test')]*len(model_numbers),
                                        keys = model_numbers,
                                        names = ['model'])
            select_data_is_test = grouped_data.xs(is_test_value, level ='is_test')
            select_data_is_test[['S', 'I', 'cases']] = select_data_is_test[['S', 'I', 'cases']] - mean_data_is_test[['S', 'I', 'cases']] 
            
            detrended_data[is_test_value] = select_data_is_test.reset_index()
        return pd.concat(detrended_data, names = ['is_test']).reset_index()
    if training_signals is None:
        # Select signals to be used in learning (hardcoded in helper)
        # default is use all
        signals = h.signals()
    else:
        signals = training_signals

    # Read cross-validation results
    if use_cross_val:
        w_min, c_min = h.read_cross_val(folder, use_incidence)
    else:
        # Hyperparameter values if use_cross_val==False
        # Option not used in full analysis
        w_min, c_min = hyperparameters


    # Read data (incidence)
    print("reading in parameters for each simulation")
    params_df = pd.read_csv(folder+"/simulation_parameters.csv")
    
    print("reading in input parameters")
    params_sim = h.simulator_args()
    if calculate_ews:
        # Read in time series
        if data_detrend:
            
            df_original = pd.concat(
                [pd.read_csv(folder_simulation_output+"/data_" + str(i) + ".csv")
                .assign(model=i).assign(is_test=int(params_df.loc[i, "R0_f"] == 0))
                for i in params_df.index.values])
            df = realisation_detrending(df_original)
        else:
            df = pd.concat(
                [pd.read_csv(folder_simulation_output+"/data_" + str(i) + ".csv")
                .assign(model=i).assign(is_test=int(params_df.loc[i, "R0_f"] == 0))
                for i in params_df.index.values]) 
            df_original = df.copy()
        # calculate EWS
        edf = h.get_ews(df, params_df, agg=aggregation_period, wtime=w_min,
                        mv_method=mv_method, use_incidence=use_incidence, data_detrend = data_detrend,
                        original_data=df_original, center = center)
        edf.to_csv(folder + "/ews_data" +
                   h.incidence_filepath(use_incidence) + ".csv")
        
        #normalise EWS
        if normalise_ews:
            def calculate_zscore(dataframe):
                '''
                Function for normalising the data in real-time
                
                For a time-series x(t) the normalised data y(t) = (x(t) - mean[x(1:t)])/std[x(1:t)]
                '''
                rolling_mean = dataframe.expanding().mean()
                rolling_std = dataframe.expanding().std(ddof=0)
                normalise_stats = (dataframe - rolling_mean)/rolling_std
                return normalise_stats
            zscoredata = [calculate_zscore(ews_data[ews_data['model']==x][stats]) for x in ews_data.model.unique()]
            dfzscore = pd.concat(zscoredata)
            dfzscore[['model','is_test','R0','Time']] = ews_data[['model','is_test','R0','Time']]
            dfzscore.to_csv(folder+'zscore_ews_data' +
                   h.incidence_filepath(use_incidence) + ".csv")

    else:
        if use_normalise:
            edf = pd.read_csv(folder + "/zscore_ews_data" +
                          h.incidence_filepath(use_incidence) + ".csv", index_col=0)
        else:
            edf = pd.read_csv(folder + "/ews_data" +
                          h.incidence_filepath(use_incidence) + ".csv", index_col=0)
        

    # Get data points used for training (last 10 years)
    # ews_data = edf[edf.Time.between(10 * 365, 20*365)].copy()
    ews_data = edf[edf.Time.between((params_sim["BurnTime"]+params_sim['Time']-200),
                                    (params_sim["BurnTime"]+params_sim["Time"]))].copy()
    # Drop NAs
    ews_data = ews_data[~ews_data.isin([np.inf, -np.inf, np.nan]).any(axis=1)]
    ews_data = ews_data[~ews_data[signals].isna().any(axis=1)]

    # Train classifier
    print(ews_data.head())
    print("logistic regression starting")
    coefs, lr_clf = elr.ews_logistic_regression(ews_data[ews_data["Time"] > 
                                                         (params_sim['BurnTime'])],
                                                standardise=True,
                                                signals=signals, do_pca=False,
                                                penalty="l1", C=c_min,
                                                solver="liblinear")
    c2 = pd.Series(coefs)
    print(coefs)
    ews_data["all"] = ews_data[signals].dot(c2[signals]) \
                        + c2["intercept"]


    # Calculate AUC through time for each EWS and for the combination 
    auc_dict = {signal: ews_data.groupby("Time").apply(
                            lambda x: roc_auc_score(x["is_test"], x[signal]))
                for signal in signals + ["timeseries"]}
    auc_dict["all"] = ews_data.groupby("Time").apply(
        lambda x: roc_auc_score(x["is_test"], x["all"]))

    # ROC for each individual EWS and all combined
    fpr_dict = {}
    tpr_dict = {}
    thresh_dict = {}
    roc_auc_dict = {}
    df_c_dict = {}
    for signal in signals + ["timeseries", "all"]:
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
    pd.Series(df_c_dict).to_csv(folder + "/optimum_thresholds" +
                                h.incidence_filepath(use_incidence) +
                                h.normalise_filepath(use_normalise) + ".csv")

    pd.Series(coefs).to_csv(folder + "/ews_weights" +
                            h.incidence_filepath(use_incidence) +
                             h.normalise_filepath(use_normalise) +".csv")

    pd.DataFrame(auc_dict).to_csv(folder + "/auc_time_series" +
                                  h.incidence_filepath(use_incidence) +
                                   h.normalise_filepath(use_normalise) +".csv")

    if save_df:
        ews_data.to_csv(folder + "/ews_data_test" +
                   h.incidence_filepath(use_incidence) +
                    h.normalise_filepath(use_normalise) +".csv")

