import classifier_training as ct
import cross_validation as cv
import sys
sys.path.append("..")
import helper as h
import gillespie_SIS as SIS
import pandas as pd

f_root = "../../data/"
f_sim = '/storage/'
params = h.simulator_args()
print("create simulation dataset")

# Either open or create simulation csv file (each row in file gives the initial R0 and final R0 for each simulation)
try:
    simulation_parameters = pd.read_csv(f_root+'/simulation_parameters.csv')
except:
    SIS.create_simulation_parameters(folder = f_root,
                                 params= h.simulator_args())

aggregation_period=5
use_incidence = 1
run_cross_validation = False
use_normalise =1 
if run_cross_validation:
    cv.run_cross_validation(use_incidence=use_incidence,
                       folder=f_root,
                       folder_simulation_output = f_sim,
                       aggregation_period=aggregation_period,
                       use_parallel=False,
                       n_jobs=1
                       )
    
# Run Classifier training
# Returns CSV file for EWSs, coefficients from logisitic regression, optimal thresholds and AUC score
ct.run_classifier_training(use_incidence=use_incidence,
                        use_cross_val=run_cross_validation,
                        calculate_ews=True,
                        normalise_ews=True,
                        use_normalise = use_normalise,
                        folder=f_root,
                        folder_simulation_output = f_sim,
                        save_df=True,
                        aggregation_period=aggregation_period,
                        training_signals = None,
                        hyperparameters = (50, 0.0001),
                        mv_method='uniform',
                        data_detrend = True,
                        center = False)

