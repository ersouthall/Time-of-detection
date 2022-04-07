function Run_MLE_changepoint(run, step)
%INPUTS
%run: simulation number (1:500)
%step: select length of time-series, step is the sample frequency (of the original data of length 500)

% Read data and get step 
tic
FixData=dlmread('../fix_data.csv'); 
FixChangeData= dlmread('../fix_change_data13.csv');
ExtData=dlmread('../ext_data.csv'); 

save_output = '../data/results/MLE';
Time = 500;
FixData = FixData(2:end, 2:end);
ExtData = ExtData(2:end, 2:end);
FixChangeData = FixChangeData(2:end, 2:end);

sizeT = Time/step;
FixData_T = FixData(:, 1:step:end);
ExtData_T = ExtData(:, 1:step:end);
FixChangeData_T = FixChangeData(:, 1:step:end);

% Realisation detrending
detrend_FixData = FixData_T - mean(FixData_T, 1);
detrend_ExtData = ExtData_T - mean(ExtData_T, 1);
detrend_FixChangeData = FixChangeData_T - mean(FixChangeData_T,1);

% Matrices are used to form the constraint when using fmincon on the optimiser (find the minimum)
% This means that Ax <= 0, where x = (sigma1, sigma2)
% For new cases, we expect sigma1 > sigma2 >=0, indicating A = [0, -1; -1, 1]
% For prevalence data, we expect sigma2 > sigma1 >=0, indicating A = [1, -1; -1, 0]

A_cases = [0, -1; -1, 1];
A_prev = [1, -1; -1, 0];

% Set the proportion of the data to consider. For implementing up-to the bifurcation, set p =0.8
prop = 1;
up_to_bif = prop*sizeT;

% Run MLE
dataext = detrend_ExtData(run,:);
data = MLE_alternate_hypothesis_tau(dataext, step, sizeT,prop, A_cases);
save(strcat(save_output,'/ext_mle_results_T_',string(up_to_bif),'_run_',string(run), '.mat'), ...
    'data');

datafix = detrend_FixData(run,:);
data = MLE_alternate_hypothesis_tau(datafix, step, sizeT,prop, A_cases);
save(strcat(save_output,'/fix_mle_results_T_',string(up_to_bif),'_run_',string(run), '.mat'), ...
    'data');

datafixC = detrend_FixChangeData(run,:);
data = MLE_alternate_hypothesis_tau(datafixC, step, sizeT, prop, A_cases);
save(strcat(save_output,'/fixCHANGE_mle_results_T_',string(up_to_bif),'_run_',string(run), '.mat'), ...
    'data');

toc
end







