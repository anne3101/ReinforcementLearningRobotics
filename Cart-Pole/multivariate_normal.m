function fx = multivariate_normal(x, mu, sigma)
%Multivariate normal (Gaussian) distribution
%Inputs:
%   x: sample
%   mu: mean vector
%   sigma:covariance matrix
%Output: 
%   fx: value of normal distribution evaluated  at sample x

%Ensure positive definiteness of covariance matrix Sigma
veig=eig(sigma);
while min(veig)<0.000001
    reg_coef = 0.04;
    nn=0; 
    variance = trace(sigma)/(nn+1);
    variance = max(variance,0.01);
    sigma = sigma + reg_coef * variance^2 * eye(size(sigma));
    veig = eig(sigma);
end

%calculate output value of normal distribution
p = length(x);
factor = 1 / (sqrt((2 * pi) ^ p * det(sigma)));
fx = factor * exp(-0.5 * (x - mu)' * inv(sigma) * (x - mu));
end