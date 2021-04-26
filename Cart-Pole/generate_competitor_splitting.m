function competitors = generate_competitor_splitting(competitors, idx_best, params, x, y)
%% split winning competitor in three and generate new competitors
%Inputs: 
%   competitors: struct of competitors to be managed
%   idx_best: index of winning competitor
%   params: struct containing hyperparameters
%Output: 
%   competitors: updated struct of competitors

%set hyperparameters
num_g = params.number_gaussians;

%% get dimension along which the variance is largest
[sz, ~, ~] = size(competitors(idx_best).sigma);
variance = zeros(sz);
for i = 1:num_g
    variance = variance .* competitors(idx_best).sigma(:, :, i); 
end
highest_var = -inf;
for i = 1:sz
    if variance(i, i) > highest_var
        highest_var = variance(i, i);
        split_dim = i; %split dimension is the one along which the variance is largest
    end
end

%% split domain of winning competitor in three parts along the dimension of the largest variance
curr_domain = competitors(idx_best).domain;
length_domain = curr_domain(split_dim, 2) - curr_domain(split_dim, 1);

domain_1 = curr_domain; 
domain_2 = curr_domain; 
domain_3 = curr_domain;

%length of each new domain = half of length of old domain (along split
%dimension)
domain_1(split_dim, :) = [curr_domain(split_dim, 1) (curr_domain(split_dim, 1) + 0.5 * length_domain)];
domain_2(split_dim, :) = [(curr_domain(split_dim, 1) + 0.25 * length_domain) (curr_domain(split_dim, 1) + 0.75 * length_domain)];
domain_3(split_dim, :) = [(curr_domain(split_dim, 1) + 0.5 * length_domain) curr_domain(split_dim, 2)];

%% add the new competitors
larger = (sum(domain_1(:,1) <= x) == length(x));
smaller = (sum(domain_1(:,2) >= x) == length(x));
in_domain = larger && smaller;
new_idx = length(competitors) + 1;
if in_domain
    competitors(new_idx) = initialize_competitor(domain_1, params, [x; y]);
else
    competitors(new_idx) = initialize_competitor(domain_1, params, 0);
end 

larger = (sum(domain_2(:,1) <= x) == length(x));
smaller = (sum(domain_2(:,2) >= x) == length(x));
in_domain = larger && smaller;
new_idx = new_idx + 1;
if in_domain
    competitors(new_idx) = initialize_competitor(domain_2, params, [x; y]);
else
    competitors(new_idx) = initialize_competitor(domain_2, params, 0);
end 

larger = (sum(domain_3(:,1) <= x) == length(x));
smaller = (sum(domain_3(:,2) >= x) == length(x));
in_domain = larger && smaller;
new_idx = new_idx + 1;
if in_domain
    competitors(new_idx) = initialize_competitor(domain_3, params, [x; y]);
else
    competitors(new_idx) = initialize_competitor(domain_3, params, 0);
end 

end