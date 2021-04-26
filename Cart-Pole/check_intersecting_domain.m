function [domain_existent, inters_domain] = check_intersecting_domain(competitors, idx_best, x, y, params)
%% gets intersection of the winning competitor's domain with the domain of the competitor that showed the least prediction errror
%Inputs: 
%   competitors: struct of competitors who should be compared
%   idx_best: index of the winning competitor
%   (x, y): sample
%   params: struct that gives the hyperparameters
%Outputs: 
%   domain_existent: Boolean, true if intersecting domain already exists as
%                   domain of another competitor
%   inters_domain: intersecting domain

%% get competitor with least prediction error in whose domain the sample is
e_pred_least = inf;
idx_least_error = 0;

%check competitors with indices lower than idx_best
for comp = 1:length(competitors(1:idx_best-1))
    larger = (sum(competitors(comp).domain(:,1) <= x) == length(x));
    smaller = (sum(competitors(comp).domain(:,2) >= x) == length(x));
    in_domain = larger && smaller;
    
    if in_domain 
        %sample is covered by domain, get prediction error and compare it 
        [y_pred, ~] = predict_y_variance(x, competitors(comp), params);
        e_pred_comp = (y_pred - y)^2;
        if e_pred_comp < e_pred_least
            e_pred_least = e_pred_comp;
            idx_least_error = comp;
        end
    end
end

%check competitors with indices higher than idx_best
for comp = idx_best+1:length(competitors)
    larger = (sum(competitors(comp).domain(:,1) <= x) == length(x));
    smaller = (sum(competitors(comp).domain(:,2) >= x) == length(x));
    in_domain = larger && smaller;
    
    if in_domain
        %sample is covered by domain, get prediction error and compare it 
        [y_pred, ~] = predict_y_variance(x, competitors(comp), params);
        e_pred_comp = (y_pred - y)^2;
        if e_pred_comp < e_pred_least
            e_pred_least = e_pred_comp;
            idx_least_error = comp;
        end
    end
end

%% %get intersection of winning competitor and competitor with least prediction error

if idx_least_error == 0
    %only the winning competitor is active in this domain -> will be split,
    %set domain_existent to true
    domain_existent = true;
    inters_domain = competitors(idx_best).domain;
else
    %get intersecting domain
    domain_existent = false;
    domain1 = competitors(idx_best).domain;
    domain2 = competitors(idx_least_error).domain;
    
    [n,m] = size(domain1);
    inters_domain = zeros(n,m);
    
    for i = 1:n
        entry_1 = max(domain1(i, 1), domain2(i, 1));
        entry_2 = min(domain1(i, 2), domain2(i, 2));
        
        if entry_2 > entry_1
            %domains did not intersect! 
            %split domain of winning competitor 
            domain_existent = true;
        end
        
        inters_domain(i, 1) = entry_1; 
        inters_domain(i, 2) = entry_2; 
    end
    
    %check if intersecting domain already exists 
    for comp = 1:length(competitors)
        if inters_domain == competitors(comp).domain
            domain_existent = true; %in this case the winning competitor's domain should be split
        end
    end
    
end

end
