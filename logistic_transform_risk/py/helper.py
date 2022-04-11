import pandas as pd
import ews
from joblib import Parallel, delayed
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats


# Signals used in analysis
def signals():
    return [ "index_of_dispersion", "autocorrelation",
            "coefficient_of_variation", "kurtosis", "skewness",
            "standard_deviation", "ac2", "autocovariance"]  # , "sd_convexity"]


def sig_labels():
    return {"ac2": "Autocorrelation (lag 2)",
            "autocorrelation": "Autocorrelation (lag 1)",
            "mean": "Mean",
            "index_of_dispersion": "Index of dispersion",
            "coefficient_of_variation": "Coefficient of variation",
            "standard_deviation": "Standard deviation",
            "skewness": "Skewness",
            "kurtosis": "Kurtosis",
            "autocovariance": "Autocovariance",
            "sd_convexity": "SD convexity"}
    
def simulator_args():
    params = {"gamma": 0.2, 
          "beta0": 1,
          "N": 10000,
          "p": 1/500,
          "Time": 500,
          "BurnTime": 300,
          "realisations": 5
          }
    return params


def lhs_space():
    """
    Set up parameter space for latin hypercube sampling
    :return:
    """
    return {"N_i": (5e4, 5.0e6),
            "eta_i": (1/30, 1),
            "rp_i": (0.01, 0.5),
            "R0_i": (0.1, 0.9),
            "bb_a": (1, 5)}


def incidence_filepath(x):
    return "_incidence" if x else ""


def normalise_filepath(x):
    return "_norm" if x else ""

def read_cross_val(folder, use_incidence=True):
    """
    Get best hyperparameter combination (windowsize/half-life and penalty
    strength) from cross validation.

    :param folder: containing cross-validation results
    :param use_incidence: (boolean) Whether using case reports or incidence data
    :return: w_min, c_min
    """
    c_before_w = True

    p_performance = pd.read_csv(folder+"/k-fold-cross-validation" +
                                incidence_filepath(use_incidence)+".csv",
                                index_col=0)
    p_performance["test"] = np.sign(p_performance["auc"] + p_performance["std"]
                                    - p_performance["auc"].max()).astype(int)
    if c_before_w:
        # Minimise c_min before w_min
        c_min = p_performance[p_performance["test"] == 1]["p"].min()
        w_min = p_performance[(p_performance["test"] == 1)
                              & (p_performance["p"] == c_min)]["w"].min()
    else:
        # Minimise w_min before c_min
        w_min = p_performance[p_performance["test"] == 1]["w"].min()
        c_min = p_performance[(p_performance["test"] == 1) &
                              (p_performance["w"] == w_min)]["p"].min()

    return w_min, c_min


def get_ews(data, params_df, agg=1, wtime=200, mv_method="exp",
            use_parallel=False, nc=1, use_incidence=False, data_detrend = False,
            original_data=None, center = False):
    '''
    Calculates EWS for each simulation
    INPUT
    :param data: pandas dataframe of all simulations (column "model" contains unique number for ...
    ... each simulation)
    :param params_df: pandas dataframe of simulation parameters (each row contains information ...
    ... for each simulation)
    :param agg: int, default 4, parameter for time aggregating e.g. 4 will group the data into 4 ...
    ... week sections (monthly)
    :param wtime: int, default 200, window size for calculating ews. The number of time points in ...
    ... a moving window is given by wtime/agg
    :param mv_method: str, default "exp". Options "exp" or "uniform". Method for moving window calculation ...
    ... for EWS
    :param use_parallel: boolean, default False. Run with parallel processing or not
    :param nc: int, default 1. Number of cores for parallel processing 
    :param use_incidence: boolean default False. If True, use reported_cases/N_i, else use reported_cases
    :param data_detrend: boolean, default False. If True the data is already detrended and no detrending will be performed in this function.
    :param original_data: arr, default None. The un-detrended data can be inputted and used for the calculation of the mean
    :param center: boolean, default False.  For the moving window calculations, the calculation can either be done center = True or for right-centered window averages (center = False)
    RETURNS
    pandas dataframe which contains the calculated EWS for each simulation and column "model" ...
    ... to uniquely identify each simulation. 
    '''
    w = wtime//agg

    def mvw(x, weight="exp"):
        if weight == "exp":
            return x.ewm(halflife=w).mean()
        if weight == "uniform":
            return x.rolling(w).mean()

    def single_run_ews(df_data, df_original):
        '''
        Function aggregates simulated data by summing all cases occuring in weekly sections ...
        ... (7*agg) e.g. when agg = 4, groups into monthly/4 weekly sections
        Calculates all EWS for a single simulation using ews.get_ews()
        INPUT 
        :param sc
        sc[0]: tuple, contains ("run", "model") where "model" is the unique number ...
        ... associated to each simulation
        sc[1]: pandas dataframe of simulation ouput for model number given in sc[0][1]
        
        RETURNS
        pandas dataframe for all EWS for a single simulation, adds additional columns:
        : R0: mean R0 for each month in aggregated data
        : time: mean time for each timestamp recorded in a month
        : model: simulation number given by sc[0][1]
        : is_test: equals 1 when final R0 =1, when final R0<1 then equals 0
        '''
        
        # i = sc[0] # index, which simulation run e.g. value in [0, 9999]
        g = df_data # pandas dataframe for chosen simulation run
        goriginal = df_original #pandas dataframe, undetrended
        #group to monthly data by summing all reported cases occuring in a month 
        #get values (no longer pandas dataframe)
        # x = g.groupby(g.time // (7*agg) * 7*agg)["reported_cases"].sum()\
        #     .reset_index(drop=True).values
        datatype='I'
        #datatype='cases'
        x = g.groupby(g.time // (agg) * 1)[datatype].first()\
            .reset_index(drop=True).values
            
        x_original = goriginal.groupby(goriginal.time // (agg) * 1)[datatype].first()\
            .reset_index(drop=True).values

        # Convert to incidence defined as reported_cases/N_i
        # N_i is the intiial population size 
        params = simulator_args()
        if use_incidence:
            x = x/params["N"]
            x_original = x_original/params["N"]
            # x = x/params_df.loc[i[1], "N_i"]
        # print(x)
        e = pd.DataFrame(ews.get_ews(x, windowsize=w, ac_lag=1, se=False,
                                     kc=False,
                                     mv_method=mv_method,
                                      detrend_prev = data_detrend,
                                      original_data = x_original,
                                      center = center))
        if mv_method =='gaussian':
            mv_method_next = "uniform"
        else:
            mv_method_next = mv_method

        e["model"] = g.model.unique()[0]
        e["is_test"] = g["is_test"].iloc[0]
        gg = g.groupby(g.time // (agg) * 1) # group simulated data into months

        e["R0"] = gg["R0"].mean().values # mean value of R0 each month
        e["Time"] = gg["time"].mean().values #mean time each month (mean time stamp of each case)
        return e

    if use_parallel:
        edf_list = Parallel(n_jobs=nc)(delayed(single_run_ews)(data[data['model']==run],original_data[original_data['model']==run])
                                       for run in data.model.unique())
    else:
        edf_list = [single_run_ews(data[data['model']==run],original_data[original_data['model']==run]) for run in data.model.unique()] # for each simulation get ews

    edf = pd.concat(edf_list)
    return edf


def lr_decision_function(x, coefs, intercept):
    return np.sum(x*coefs) + intercept


def lr_emergence_risk(df):
    return 1 / (1 + np.exp(-df))