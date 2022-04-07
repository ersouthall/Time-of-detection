
function [likelihood] = LogLikelihoodNormal(data,t1, t2, mu, sigma)

likelihood = 0;
for i = t1:t2
    likelihood = likelihood+log(normpdf(data(i), mu, sigma));
end
% minus the log-likelihood, so that we can use fminsearch/fmincon
likelihood=-likelihood;
end