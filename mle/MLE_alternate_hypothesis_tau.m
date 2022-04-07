function [likelihood_results_tau, index] = MLE_alternate_hypothesis_tau(data, time, A)
% For each potential location for the change point tau, 
% calculate the loglikelihood for the alternate hypothesis
% Take the minimum loglikelihood and location of tau satisfying the min

likelihood_results_tau = zeros(1, time);
opt = optimset('Display','off');

for tau =1:(time)
    % The mean of both distributions is zero (e.g. detrended data has a mean of zero)
    changepointLogLikelihood = @(theta)LogLikelihoodNormalsigma_tau(data,tau, time, 0, theta(1), theta(2));
    [thetahatChange,LLChange,~,~,~,~,H]=fmincon(changepointLogLikelihood,[40 40], A, [0 0], ...
                                        [], [], [], [], [], opt);    
    likelihood_results_tau(tau) = LLChange;
end
[value, index] = min(likelihood_results_tau(1:time));
end
