function [results] = Likelihood_of_changepoint(data,step, sizeT, prop_upto, A)
% Settings: to not display fmincon output
opt = optimset('Display','off');
% For the figure up to the bifurcaton, need to select data (otherwise set prop_upto = 1)
up_to_bif = sizeT*prop_upto;

% Calculate the maximum likelihood of the null model (no change point, data described by N(mu=0, theta))
H0_Likelihood = @(theta)LogLikelihoodNormal(data,...
                    1, up_to_bif, 0, theta);
[thetahat,LL_H0,~,~,~,~,H]=fmincon(H0_Likelihood,40, -1, 0, ... 
    [], [], [],  [], [], opt);


% Calculate the maximum likelihood for the alternative hypothesis model (e.g. exists a change point)
[likelihood_results_tau, tau_choice] = MLE_alternate_hypothesis_tau(data, up_to_bif, A);

% Likelihood for mean and variance based off best tau
H1_Likelihood = @(theta)LogLikelihoodNormalsigma_tau(data,...
            tau_choice, up_to_bif, 0, theta(1), theta(2));
[thetahatChange,LL_H1,~,~,~,~,H]=fmincon(H1_Likelihood,[40 40], A,...
    [0 0],[], [], [], [], [], opt );    

% Likelihood ratio
D=2*(LL_H0-LL_H1);

%  Confidence intervals 
CI = confidence_interval_smoothing(likelihood_results_tau, step, up_to_bif);

results = struct('tau', tau_choice, 'LLR', D, 'LL_H0', LL_H0, ...
                  'LL_H1', LL_H1, 'CI_l', CI.CI_l, ...
                  'CI_h', CI.CI_h, 'CI_size', CI.size, 'parameters', thetahatChange);
end
