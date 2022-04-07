function [CI] = confidence_interval_smoothing(likelihood, dt, time)
% function for calculating the confidence intervals (not used in analysis of paper)
smooth_likelihood = smoothdata(likelihood, 'movmean', ceil(0.1*time) );
[value, index] = min(smooth_likelihood);

second_deriv = central_difference_2nd(smooth_likelihood, index, dt, time);
confidence_interval = [index - norminv(0.975)*sqrt(1/second_deriv) index + norminv(0.975)*sqrt(1/second_deriv)];
confidence_size = abs(confidence_interval(1)- confidence_interval(2));

CI = struct('CI_l', confidence_interval(1), 'CI_h', confidence_interval(2), 'size', confidence_size);

end
