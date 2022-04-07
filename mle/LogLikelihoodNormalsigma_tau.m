
function [likelihood] = LogLikelihoodNormalsigma_tau(data,t1,tmax, mu, sigma1, sigma2)
% Function for the likelihood when there exists a change point at t1.
% The data is described by a N(mu, sigma1) for t<= t1 and N(mu, sigma2) for t>t1
likelihood = 0;
for i = 1:t1
    likelihood = likelihood+log(normpdf(data(i), mu, sigma1));
end

for j = (t1+1):tmax
    likelihood = likelihood + log(normpdf(data(j), mu, sigma2));
end

% Minus the log-likelihood, so that we can use fminsearch/fmincon
likelihood=-likelihood;
end