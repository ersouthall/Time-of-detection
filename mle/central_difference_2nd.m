function [deriv] = central_difference_2nd(data, tau, dt, tmax)
% function for calculating the confidence intervals (not used in analysis of paper)
if (tau>1) && (tau<tmax)
    x_t = data(tau);
    x_t_plus_1 = data(tau+1);
    x_t_minus_1 = data(tau-1);
    deriv = (x_t_plus_1 - 2*x_t + x_t_minus_1)/(dt^2);
elseif (tau==1)
    x_t = data(tau);
    x_t_plus_1 = data(tau+1);
    deriv =  (- x_t + x_t_plus_1)/(dt^2);
elseif (tau==tmax)
    x_t = data(tau);
    x_t_minus_1 = data(tau-1);
    deriv =  (- x_t + x_t_minus_1)/(dt^2);
    
end
