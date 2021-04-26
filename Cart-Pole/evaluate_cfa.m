function [y, y_var] = evaluate_cfa(competitors, x, params)
%% get CFA prediction by determining the winning competitor and evaluating it
%Inputs: 
%   competitors: competitors to be evaluated
%   x: input, point in space where model should be evaluated
%   params: struct containing hyperparameters
%Outputs:
%   y: predicted value
%   idx_best: index of winning competitor

%set hyperparameters
num_g = params.number_gaussians;

%% check that x remains inside allowed ranges 
if x(1) > 6 %x
    x(1) = 6; 
elseif x(1) < -6
    x(1) = -6; 
end
if x(2) > 10 %d_x
    x(2) = 10; 
elseif x(2) < -10
    x(2) = -10; 
end
if x(3) > pi %theta
    x(3) = pi; 
elseif x(3) < -pi
    x(3) = -pi; 
end
if x(4) > 10 %d_theta
    x(4) = 10; 
elseif x(4) < -10
    x(4) = -10; 
end
if length(x) == 5
    if x(5) > 10 %action
        x(5) = 10;
    elseif x(5) < -10
        x(5) = -10;
    end
end

%% determine winning competitor
Gamma_best = -inf;
idx_best = 0;

for i=1:length(competitors)
    %get active competitors phi_x
    larger = (sum(competitors(i).domain(:,1) <= x) == length(x));
    smaller = (sum(competitors(i).domain(:,2) >= x) == length(x));
    in_domain = larger && smaller;
    
    if in_domain
        %compute value of relevance function
        Gamma = relevance_function(x, competitors(i), params);
        
        %select the winner competitor 
        if Gamma > Gamma_best
            Gamma_best = Gamma;
            idx_best = i;
        end
    end
end

%% get prediction of winning competitor 
mu_i = zeros(num_g);
variance_i = zeros(num_g, 1);
beta_numerator = zeros(num_g, 1);
beta = zeros(num_g, 1);

for i = 1:num_g
    mu_ix = competitors(idx_best).mu(1:end-1, i);
    mu_iy = competitors(idx_best).mu(end, i);
    covariance_ixx = competitors(idx_best).sigma(1:end-1, 1:end-1, i);
    covariance_iyx = competitors(idx_best).sigma(end, 1:end-1, i);
    covariance_ixy = competitors(idx_best).sigma(1:end-1,end, i);
    covariance_iyy = competitors(idx_best).sigma(end,end, i);

    variance_i(i) = covariance_iyy - covariance_iyx / covariance_ixx * covariance_ixy;
    mu_i(i) = mu_iy + covariance_iyx / covariance_ixx * (x - mu_ix);
    
    beta_numerator(i) = competitors(idx_best).alpha(i) * multivariate_normal(x, mu_ix, covariance_ixx);
end

if sum(beta_numerator) == 0 %to prevent dividing by zero
    beta_numerator = ones(num_g, 1) * 0.1;
end

y = 0; %predicted output
y_var = 0;
for i = 1:num_g
    beta(i) = beta_numerator(i) / sum(beta_numerator);
    y = y + beta(i) * mu_i(i);
    y_var = y_var + beta(i)*(variance_i(i) + (mu_i(i)-y)^2);
end

end