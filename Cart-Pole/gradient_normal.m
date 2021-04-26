function gradient = gradient_normal(x, competitors, params)
%% calculates gradient of GMM
%Inputs: 
%   x: sample (5 dimensional vector)
%   competitors: struct competitors that each have a GMM 
%   params: struct containing hyperparameters
%Outputs: 
%   gradient wrt action

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

%% get active competitors and calculate gradient
gradient_sum = 0;

for i=1:length(competitors)
    %get active competitors phi_x
    larger = (sum(competitors(i).domain(:,1) <= x) == length(x));
    smaller = (sum(competitors(i).domain(:,2) >= x) == length(x));
    in_domain = larger && smaller;
    
    if in_domain
        for g = 1:num_g
            %regularize sigma to prevent computation errors
            sigma = competitors(i).sigma(1:end-1, 1:end-1, g);
            veig=eig(sigma);
            while min(veig)<0.000001
                reg_coef = 2;
                nn=0; 
                variance = trace(sigma)/(nn+1);
                variance = max(variance,0.01);
                sigma = sigma + reg_coef * variance^2 * eye(size(sigma));
                veig = eig(sigma);
            end
            
            %calculate gradient of competitors and sum it onto the overall
            %gradient_sum
            mu = competitors(i).mu(1:end-1, g);
            grad = multivariate_normal(x, mu, sigma); 
            gradient_sum = gradient_sum - grad * sigma \ (x - mu); 
        end
    end
end

%% exclude errors in gradient computation by setting gradient to zero
if isnan(gradient_sum(5))
    gradient_sum(5) = 0;
end
if gradient_sum(5) == inf
    gradient_sum(5) = 0;
end
if gradient_sum(5) == -inf
    gradient_sum(5) = 0;
end

%only entry 5 of vector necessary for further computation
gradient = gradient_sum(5); 
end