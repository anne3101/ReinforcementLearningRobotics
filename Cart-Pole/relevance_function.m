function Gamma = relevance_function(xt, competitor, params)
%calculates relevance of competitor at sample  value xt 
%Input:
%   xt: sample x value
%   competitor: current competitor
%   params: struct of hyperparameters
%Output:
%   Gamma: calculated relevance of competitor

%set hyperparameters
num_g = params.number_gaussians;
V_i = params.V_i;

beta_numerator = zeros(num_g, 1);
for i = 1:num_g
    mu_ix = competitor.mu(1:end-1, i);
    covariance_ixx = competitor.sigma(1:end-1,1:end-1, i);    
    beta_numerator(i) = competitor.alpha(i) * multivariate_normal(xt, mu_ix, covariance_ixx);
end

if sum(beta_numerator) == 0
    beta_numerator = ones(num_g, 1) * 0.1;
end
p_i = sum(beta_numerator);

%calculate sample variance
[~, sigma_i] = predict_y_variance(xt, competitor, params);

%relevance: alpha quantile of chi^2 distribution
if competitor.n_i == 1
    deg_freedom = 1;
else
    deg_freedom = competitor.n_i * p_i * V_i - 1;
end

if deg_freedom < 1
    deg_freedom = 1;
end

alpha = 0.95;
alpha_quantile = chi2inv(alpha, deg_freedom);

Gamma = alpha_quantile/(deg_freedom * sigma_i);
end
