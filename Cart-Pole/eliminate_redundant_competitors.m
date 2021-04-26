function competitors = eliminate_redundant_competitors(competitors, params)
%% determines which competitors are redundant and eliminates them
%Inputs: 
%   competitors: struct of competitors to evaluate 
%   params: struct containing hyperparameters
%Output: 
%   competitors: updated struct of competitors

%set hyperparameters
redundance_num_points = params.redundance_num_points;
threshold_v = params.threshold_v;
threshold_r = params.threshold_r;

%% determine redundant competitors 
redundant_comp = zeros(length(competitors)); %redundant competitors will have a 1 in the list
[size_domain, ~] = size(competitors(1));

for i = 1:length(competitors)
    for j = 1:length(competitors)
        
        if i ~= j
            %check if xi subset of xj
            x_i = competitors(i).domain;
            x_j = competitors(j).domain;
            part1 = (sum(x_i(:, 1) >= x_j(:, 1)) == length(x_i));
            part2 = (sum(x_i(:, 2) <= x_j(:, 2)) == length(x_i));
            subset = part1 && part2;
            
            if subset
                %check thresholds for finite number of points in domain x_i
                samples = zeros(size_domain, redundance_num_points); %redundance_num points with size-domain x d coordinates (linearly spaced)
                for  d = 1:size_domain
                    samples(d, :) = linspace(x_i(d, 1), x_i(d, 2), redundance_num_points);
                end
                
                diff_value = threshold_v - 0.1;
                diff_relevance = threshold_r - 0.1;
                point = 1;
                
                while((diff_value < threshold_v) && (diff_relevance < threshold_r) && (point <= redundance_num_points))
                    %calculate value difference
                    [y_pred_i, ~] = predict_y_variance(samples(:, point), competitors(i), params);
                    [y_pred_j, ~] = predict_y_variance(samples(:, point), competitors(j), params);
                    diff_value = abs(y_pred_i - y_pred_j) / max(abs(y_pred_i), abs(y_pred_j));
                    
                    %calculate relevance difference
                    gamma_i = relevance_function(samples(:, point), competitors(i), params);
                    gamma_j = relevance_function(samples(:, point), competitors(j), params);
                    diff_relevance = abs(gamma_i - gamma_j) / max(abs(gamma_i), abs(gamma_j));
                    
                    if point == redundance_num_points
                        %perform last comparison of threshold and
                        %value
                        if (diff_value < threshold_v) && (diff_relevance < threshold_r)
                            %set in list that competitor i should
                            %be eliminated
                            redundant_comp(i) = 1;
                        end
                    end
                    point = point + 1;
                end
            end
        end
    end
end

%% eliminate redundant competitors

for comp = 1:length(redundant_comp)
    if redundant_comp(comp) == 1
        % competitor comp must be eliminated
        competitors_new = competitors(1:end-1);
        competitors_new(comp:end) = competitors(comp+1:end);
        competitors = competitors_new;
    end
end

end