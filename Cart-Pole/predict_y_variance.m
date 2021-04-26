function [y_pred, y_var] = predict_y_variance(xt, competitor, params) 
%predict output y and variance for competitor GMM
%Inputs: 
%   xt: sample for which output should be calculated
%   competitor: current competitor
%   params: struct containing  hpyerparameters
%Outputs: 
%   y_pred: predicted output y of competitor GMM
%   y_var:  variance of output 

%set hyperparameters
num_g = params.number_gaussians;

%initialize empty arrays
mu_i = zeros(num_g);
variance_i = zeros(num_g, 1);
beta_numerator = zeros(num_g, 1);
beta = zeros(num_g, 1);

for i = 1:num_g
    mu_ix = competitor.mu(1:end-1, i);
    mu_iy = competitor.mu(end, i);
    covariance_ixx = competitor.sigma(1:end-1,1:end-1, i);
    covariance_ixy = competitor.sigma(1:end-1,end, i);
    covariance_iyx = competitor.sigma(end,1:end-1, i);
    covariance_iyy = competitor.sigma(end,end, i);
    
    mu_i(i) = mu_iy + covariance_iyx / covariance_ixx * (xt - mu_ix);
    variance_i(i) = covariance_iyy - covariance_iyx / covariance_ixx * covariance_ixy;
    
    beta_numerator(i) = competitor.alpha(i) * multivariate_normal(xt, mu_ix, covariance_ixx);
end

if sum(beta_numerator) == 0 %prevent dividing by 0
    beta_numerator = ones(num_g, 1) * 0.1;
end

y_pred = 0; %predicted output
y_var = 0; %predicted variance of output

for i = 1:num_g
    beta(i) = beta_numerator(i) / sum (beta_numerator);
    y_pred = y_pred + beta(i) * mu_i(i);
    y_var = y_var + beta(i) * (variance_i(i) + (mu_i(i)-y_pred)^2);
end

end