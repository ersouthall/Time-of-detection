import itertools
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import ews_logistic_regression as elr
import sys
sys.path.append("..")
import helper as h


def run_cross_validation(use_incidence, folder, folder_simulation_output, aggregation_period=4,
                         use_parallel=False, n_jobs=1):
    '''
    Calculates pandas dataframe containing auc (mean and std), window size (w) and ...
    ... regularisation strength of Lasso Regression (p). 
    Trains lasso regression on 1000 simulations and tests on 9000 simulations. Does this for 
    10 chunks of simulations and takes mean and std

    INPUT
    :param use_incidence: boolean, if true takes reported_cases/Ni, else reported_cases
    :param folder: str, location of parameter set for each simulation
    :param folder_simulation_output: str, location of simulated data
    :param aggregation_period: int, default 4, time period for aggregation e.g. agg=4 groups ...
    ... the data into 4 week sections (e.g. monthly)
    :param use_parallel: boolean, default False. If true, uses parallel processing 
    :param n_jobs: int, default 1. Number of cores for parallel processing
    RETURN
    saves pandas dataframe to:
    folder+"/k-fold-cross-validation" + h.incidence_filepath(use_incidence) + ".csv"
    
    h.incidence_filepath(True): returns "incidence_"
    '''
    # Select signals to be used in learning:
    signals = h.signals()

    # Read in parameters:
    params_df = pd.read_csv(folder+"/simulation_parameters.csv")
    n_models = params_df.shape[0]

    # Read in time series and combines all simulations together 
    # adds two new columns to dataframe: model (corresponding to simulation number) 
    # is_test (equal to 1 when R0_f =1, equal to 0 when R0_f is not 1)
    df = pd.concat(
        [pd.read_csv(folder_simulation_output+"/data_" + str(i) + ".csv")
         .assign(model=i).assign(is_test=int(params_df.loc[i, "R0_f"] == 1.0))
         for i in params_df.index.values])
    print("Read in files: ok")

    # Cross-validation
    def chunks(l, n):
        """Yield successive n-sized chunks from l.
        Input
        : l: array
        : n: size of chunk """
        for i in range(0, len(l), n):
            yield l[i:i + n]


    np.random.seed(3658042)
    n_chunks = 10
    ## select half of the simulations e.g. first 5000 simulations
    rnd_models = np.random.permutation(params_df.index.values[:(n_models//2)])
    ## size of each chunk will be len(rnd_models)//n_chunks
    ## get an array of size n_chunnk by len(rnd_models)//n_chunks
    ## i.e. each row is a single chunk of size len(rnd_models)//n_chunks containing ...
    ## ... all the randomally selected simulation IDs
    chunk_arr = np.array([chunk for chunk in chunks(rnd_models,
                                                    len(rnd_models)//n_chunks)])

    p_str = [100, 10, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001] #regularize strength,...
    #... smaller values indicate stronger strength 
    w_str = [52, 104, 156, 208, 260, 312]
    plist = []
    auc_list = []

    for w in w_str:
        ### for each windowsize w, calculate early warning signals
        ews_data = h.get_ews(df, params_df, agg=aggregation_period,
                             wtime=w, mv_method="exp",
                             use_incidence=use_incidence)
        ews_data = ews_data[ews_data.Time > 10 * 365] # remove the first 10 years
        ews_data = ews_data[~ews_data.isin([np.inf, -np.inf, np.nan])
            .any(axis=1)]
        ews_data = ews_data[~ews_data[signals].isna().any(axis=1)]
        ews_g = ews_data.groupby("model")
        print("Calculate ews: ok")

        def get_auc_for_chunk(i, p):
            ''' 
            Uses Lasso Regression with a training dataset of EWS and binary observations ...
            ... is_test that records 1 if emerging and 0 if non-emerging
            Returned weights and y-intercept from regression are used to calculate weighted sum of EWS
            ... on testing dataset
            Area under the curve (ROC curve) returned (calculated on test data and test observations)
            
            INPUT
            : i: int, index for chunk_arr (i.e input can be from 0 to n_chunk )
            : p: positive float, regularize strength used in Lasso Regression
            RETURNS
            pandas series containing:
            : w: window size used in getting the EWS
            : p: regularize strength for Lasso Regression
            : auc: area under the curve score on the test data with weights and y-intercept ...
            ... produced from Lasso Regression
            '''
            chunk = chunk_arr[i] # shape 0 by len(rnd_models)//n_chunks
            # add n_models//2 onto all simulation indices, to double size of models
            models = np.concatenate((chunk, chunk + n_models // 2))
            
            #filters groupby("model") of ews_g to select train and test
            train_models = ews_g.filter(lambda x: x.name not in models)
            test_models = ews_g.filter(lambda x: x.name in models)
            # print("test/train ok")

            coefs, lr_clf = elr.ews_logistic_regression(train_models,
                                                        standardise=True,
                                                        signals=signals,
                                                        do_pca=False,
                                                        penalty="l1",
                                                        C=p,
                                                        solver="liblinear")

            c2 = pd.Series(coefs)
            # calculate weighted sum plus the intercept
            tdf = test_models[signals].dot(c2[signals]) + c2["intercept"] 
            auc = roc_auc_score(test_models["is_test"], tdf)

            print(w, i, p)
            return pd.Series({"p": p, "w": w, "auc": auc}) #returns strength, window and auc

        #iterate between all chunks and regularize strength p_str
        loop_pars = list(itertools.product(np.arange(n_chunks), p_str)) 

        if use_parallel:
            auc_list += Parallel(n_jobs=n_jobs)(delayed(get_auc_for_chunk)(*par)
                                                for par in loop_pars)
        else:
            auc_list += [get_auc_for_chunk(*par) for par in loop_pars]

    auc_cval = pd.concat(auc_list, axis=1).transpose()
    # mean and std over 10 chunks
    auc_g = auc_cval.groupby(["p", "w"]).agg({"auc": ["mean", "std"]})\
        .reset_index()
    auc_g.columns = ["p", "w", "auc", "std"]
    p_performance = auc_g

    p_performance.to_csv(folder+"/k-fold-cross-validation" +
                         h.incidence_filepath(use_incidence)
                         + ".csv")

  