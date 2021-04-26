function competitors = update_CFA(competitors, x, y, iterations, params)
%% updates competitors of  CFA
%Inputs 
%   competitors: competitors that should be updated
%   (x,y): input sample, used as updating reference
%   iterations: current training iteration
%   params: struct containing hyperparameters
%Output: 
%   competitors: updated competitor

%set hyperparameters
threshold_n = params.threshold_n;
threshold_error = params.threshold_error; 
i_elim = params.i_elim;
%% check that x remains inside allowed ranges 
if x(1) > 6
    x(1) = 6; 
elseif x(1) < -6
    x(1) = -6; 
end
if x(2) > 10 
    x(2) = 10; 
elseif x(2) < -10
    x(2) = -10; 
end
if x(3) > pi
    x(3) = pi; 
elseif x(3) < -pi
    x(3) = -pi; 
end
if x(4) > 10 
    x(4) = 10; 
elseif x(4) < -10
    x(4) = -10; 
end
if length(x) == 5
    if x(5) > 10
        x(5) = 10;
    elseif x(5) < -10
        x(5) = -10;
    end
end

%% update GMM params of active competitors and select winning competitor
n_min = inf;
Gamma_best = -inf;
idx_best = 0;

for i=1:length(competitors)
    larger = (sum(competitors(i).domain(:,1) <= x) == length(x));
    smaller = (sum(competitors(i).domain(:,2) >= x) == length(x));
    in_domain = larger && smaller;
    
    if in_domain    
        % update parameters of GMM of each active competitor using EM
        competitors(i) = update_gmm_parameters(competitors(i), x, y, params);
        
        % get minimum number of samples n_min
        if competitors(i).n_i < n_min
            n_min = competitors(i).n_i;
        end
        
        %compute value of relevance function
        Gamma = relevance_function(x, competitors(i), params);
        
        %select the winner competitor in x, phi_w(x)
        if Gamma > Gamma_best
            Gamma_best = Gamma;
            idx_best = i;
        end
    end 
end

%% manage the competitors

%get prediction error of winnning competitor
[y_pred, ~] = predict_y_variance(x, competitors(idx_best), params);
e_pred = (y_pred - y)^2;

if n_min > threshold_n && e_pred > threshold_error
    %check if intersecting domain exists
    [domain_existent, inters_domain] = check_intersecting_domain(competitors, idx_best, x, y, params);
    
    if domain_existent
        %generate new competitor by splitting
        competitors = generate_competitor_splitting(competitors, idx_best, params, x, y);
    else
        %generate new competitor with intersecting domain
        new_idx = length(competitors) + 1;
        competitors(new_idx) = initialize_competitor(inters_domain, params, [x; y]);
    end   
end

if mod(iterations, i_elim) == 1
    % elimate redundant competitors
    competitors = eliminate_redundant_competitors(competitors, params);
end
end