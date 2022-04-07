function [R, log_R] = shiryaev_roberts_stat(f_pre_var, g_post_var, data, length_time)
% likelihood f()/g() and log likelihood
L = normpdf(data, 0, g_post_var)./normpdf(data, 0, f_pre_var);
log_L = log(normpdf(data,0,g_post_var))-log(normpdf(data, 0, f_pre_var));
R(1) = L(1);
log_R(1) = log_L(1);
% recursive relationship for the Shiryaev–Roberts (SR) Procedure 
for k = 2:length_time
    R(k) = (1+R(k-1))*L(k);
    log_R(k) = log_L(k) + log(1+R(k-1));
end
end