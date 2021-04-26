function [competitor] = initialize_competitor(domain, params, input_mu) %create new competitor with num_g Gaussians in domain
%% initialize new competitor with num_g Gaussians in domain
%Inputs: 
%   domain: domain of new competitor 
%   params: struct containing hyperparameters
%   input_mu: the mean of one Gaussian
%Output: 
%   struct of new competitor

%set hyperparameters
num_g = params.number_gaussians;

%initialize domain 
competitor.domain = domain;
[sz, ~] = size(domain);

%initialize number of samples observed in the domain
competitor.n_i = 0;

%% initialize parameters of the Gaussians
%randomly sample the locations of the means of the Gaussians
xt = zeros(sz, num_g); 
for d = 1:sz
    xt(d, :) = domain(d,1) + (domain(d, 2) - domain(d, 1)) .* rand(1, num_g); 
end

%initialize empty parameters of Gaussians
mu = zeros(sz + 1, num_g);
sigma = zeros(sz + 1, sz + 1, num_g);
W = ones(1, num_g);
alpha = ones(1, num_g) * 0.1;
X = zeros(sz + 1, num_g);
XX = zeros(sz + 1, sz + 1, num_g);

for i = 1:num_g
    %randomly sample the mean values inside given ranges
    if sz == 4 %comptitor for policy
        ini_value_mu = 1 / (xt(3, i)) * 5;
    else %competitor for Q value
        ini_value_mu = 1/ xt(3, i) * 50;
    end
    %initialize parameters of Gaussians
    
    %make sure the values remain inside the allowed ranges
    if ini_value_mu > 10
        ini_value_mu = 10;
    elseif ini_value_mu < -10
        ini_value_mu = 10;
    end
    mu(:, i) = [xt(:,i) ; ini_value_mu];

    if (length(input_mu) ~= 1) && (i == num_g)
        %set the mean of one Gaussian to be input_mu
        mu(:, i) = input_mu; 
    end

    sigma(:, :, i) = 1000 * eye(sz + 1);
    
    %initialize values for online EM 
    X(:, i) = mu(:, i); 
    XX(:, :, i) = sigma(:, :, i) - mu(:, i) * mu(:, i)';
end

competitor.alpha = alpha; 
competitor.mu = mu; 
competitor.sigma = sigma; 
competitor.W = W; 
competitor.X = X; 
competitor.XX = XX;
end