% read data (from csv files)
FixData=readmatrix(strcat('../data/Fix_data.csv')); 
FixCHANGEData=readmatrix(strcat('../data/FixChange_data.csv')); 
ExtData=readmatrix(strcat('../data/Ext_data.csv')); 

realisations = 500;
Time = 500;
FixData = FixData(2:end, 2:end);
FixCHANGEData = FixCHANGEData(2:end, 2:end);
ExtData = ExtData(2:end, 2:end);

% Select length of time-series
% If step is the sample frequency (taking every 5th data point)
% Paper uses step: 2 (sizeT = 250), 5 (100), 10 (50), 25 (20)
step = 5;
sizeT = Time/step;

% up_to_bif =sizeT*0.9;
FixData_T = FixData(:, 1:step:Time);
FixCHANGEData_T = FixCHANGEData(:, 1:step:Time);
ExtData_T = ExtData(:, 1:step:Time);

% Realisation detrending
detrend_FixData = FixData_T - mean(FixData_T, 1);
detrend_FixCHANGEData = FixCHANGEData_T - mean(FixCHANGEData_T, 1);
detrend_ExtData = ExtData_T - mean(ExtData_T, 1);

% Option: to take the log of the Shiryaev–Roberts (SR) Procedure 
opt = optimset('Display','off');
RR_fix = zeros(realisations, sizeT);
log_RR_fix = zeros(realisations, sizeT);

RR_fixCHANGE = zeros(realisations, sizeT);
log_RR_fixCHANGE = zeros(realisations, sizeT);

RR_ext = zeros(realisations, sizeT);
log_RR_ext = zeros(realisations, sizeT);

% Choice of variance for the Normal distributions f(0, sigma1) and g(0, sigma1)
sigma1 = 34.0002; %40
sigma2 = 2.4292; %10

%Save data
for (run = 1:realisations)
    Fixdata_single = detrend_FixData(run,:);
    [FixR, FixlogR] = shiryaev_roberts_stat(sigma1, sigma2, Fixdata_single, sizeT);
    RR_fix(run, :) = FixR; 
    log_RR_fix(run,:) = FixlogR;

    FixCHANGEdata_single = detrend_FixCHANGEData(run,:);
    [FixCHANGER, FixCHANGElogR] = shiryaev_roberts_stat(sigma1, sigma2, FixCHANGEdata_single, sizeT);
    RR_fixCHANGE(run, :) = FixCHANGER; 
    log_RR_fixCHANGE(run,:) = FixCHANGElogR;
    
    Extdata_single = detrend_ExtData(run,:);
    [ExtR, ExtlogR] = shiryaev_roberts_stat(sigma1, sigma2, Extdata_single, sizeT);
    RR_ext(run, :) = ExtR; 
    log_RR_ext(run,:) = ExtlogR;
end
save(strcat('../data/quickest_detection/log_quickest_detection_FIX', ...
    string(sizeT)), 'log_RR_fix')

save(strcat('../data/quickest_detection/log_quickest_detection_FIXCHANGE', ...
    string(sizeT)), 'log_RR_fixCHANGE')

save(strcat('../data/quickest_detection/log_quickest_detection_EXT', ...
    string(sizeT)), 'log_RR_ext')
