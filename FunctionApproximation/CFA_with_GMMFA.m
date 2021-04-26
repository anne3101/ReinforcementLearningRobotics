classdef CFA_with_GMMFA
    
    properties
        threshold_mse
        forget_a
        forget_b
        threshold_error
        threshold_n % threshold for minimum number of sample
        i_elim % number of sample when to check for redundant competitors
        initial_num_competitors
        V_i %empirically defined volume of sample neighborhood
        alpha % Quantile of Chi^2 distribution
        %w_new %predefined weight for initialization of covariance matrix of Gaussian
        redundance_check_num_points
        threshold_v
        threshold_r
    end
    
    methods
        
        function CFA(obj)
            initializeMsePlot(obj);
            
            % Initialise degenerate function using set of competitors
             
            % Initialise GMMs of each competitor with set of 2 Gaussians
            domains = linspace(-5, 5, obj.initial_num_competitors+1);
            overlap = 8 / (2*obj.initial_num_competitors);
            for i = 1:obj.initial_num_competitors
                if i == 1
                    domain = [(domains(i)) (domains(i+1)+overlap)];
                elseif i == obj.initial_num_competitors
                    domain = [(domains(i)-overlap) (domains(i+1))];
                else
                    domain = [(domains(i)-overlap) (domains(i+1)+overlap)];
                end
                competitors(i) = initialize_competitor(obj, domain);
            end
            
            mse = obj.threshold_mse + 1;
            episodes=1;
            while (mse > obj.threshold_mse && episodes < 200)
                for domain=["forward", "backward"]
                    %get observation for domain (x, y) 
                    [x, y] = get_observation(obj, domain); %swap of observations
                    
                    for sample = 1:length(x)
                        
                        n_min = inf;
                        Gamma_best = -inf;
                        for i=1:length(competitors)
                            
                            %get active competitors phi_x and update parameters
                            if competitors(i).domain(1)<=x(sample) && competitors(i).domain(2) >= x(sample)
                                
                                % update parameters of GMM of each active competitor using EM
                                competitors(i) = update_gmm_parameters(obj, competitors(i), x(sample), y(sample));
                                
                                % get minimum number of samples n_min
                                if competitors(i).n_i < n_min
                                    n_min = competitors(i).n_i;
                                end
                                
                                %compute value of relevance function
                                Gamma = relevance_function(obj, x(sample), y(sample), competitors(i));
                                
                                %select the winner competitor in x, phi_w(x)
                                if Gamma > Gamma_best
                                    Gamma_best = Gamma;
                                    idx_best = i;
                                end
                            end
                     
                        end                   
                        
                        % calculate approximation error e=(y- phi_w(x))^2
                        [y_pred, ~] = gmm(obj, x(sample), competitors(idx_best));
                        e_pred = (y_pred - y(sample))^2;
                        
                         if n_min > obj.threshold_n && e_pred>obj.threshold_error
                             %check if intersecting domain exists
                             [domain_existent, inters_domain] = check_intersecting_domain(obj, competitors, idx_best, x(sample), y(sample));
                             
                             if domain_existent
                                 %generate new competitor with splitting
                                 competitors = generate_competitor_splitting(obj, competitors, idx_best); 
                             else
                                 %generate new competitor with intersecting
                                 %domain
                                 new_idx = length(competitors) + 1;
                                 competitors(new_idx) = initialize_competitor(obj, inters_domain);
                             end
                                 
                         end
                         
                         if mod(sample, obj.i_elim) == 1
                             % elimate redundant competitors from phi_x
                             competitors = eliminate_redundant_competitors(obj, competitors);
                         end
                    end
                    
                    %get MSE of current domain
                    mse = 0;
                    y_pred = zeros(size(x));
                    y_var = zeros(size(x));
                   
                    for sample=1:length(x)
                        %get prediction of the best competitor
                        for i=1:length(competitors)                            
                            %get active competitors
                            Gamma_best = - inf;
                            if competitors(i).domain(1)<=x(sample) && competitors(i).domain(2) >= x(sample)
                                                                
                                %compute value of relevance function
                                Gamma = relevance_function(obj, x(sample), y(sample), competitors(i));
                                
                                %select the winner competitor in x, phi_w(x)
                                if Gamma > Gamma_best
                                    Gamma_best = Gamma;
                                    idx_best = i;
                                end
                            end
                        end
                        [y_pred(sample), y_var(sample)] = gmm(obj, x(sample), competitors(idx_best)); 
                       
                        %get prediction error
                        mse = mse + (y_pred(sample) - y(sample))^2;
                    end
                    plot_mse(obj, mse, episodes);
                    episodes = episodes + 1;
                end
            end
            %plot last prediction 
            plot_funcapprox(obj, y_pred, y_var, y)
            disp("Final number of competitors: ")
            disp(length(competitors))
        end
        
        function [domain_existent, inters_domain] = check_intersecting_domain(obj, competitors, idx_best, x, y)
            % get competitor with least prediction error 
            % missing: sample must be in competitor domain
            e_pred_least = inf;
            idx_least_error = 0;
            for comp = 1:length(competitors(1:idx_best-1))
                if (competitors(comp).domain(1) <= x) && (competitors(comp).domain(2) >= x)
                    [y_pred, ~] = gmm(obj, x, competitors(comp));
                    e_pred_comp = (y_pred - y)^2;
                    if e_pred_comp < e_pred_least
                        e_pred_least = e_pred_comp;
                        idx_least_error = comp;
                    end
                end
            end
            for comp = idx_best+1:length(competitors)
                if (competitors(comp).domain(1)<=x) && (competitors(comp).domain(2) >= x)
                    [y_pred, ~] = gmm(obj, x, competitors(comp));
                    e_pred_comp = (y_pred - y)^2;
                    if e_pred_comp < e_pred_least
                        e_pred_least = e_pred_comp;
                        idx_least_error = comp;
                    end
                end
            end
            
            if idx_least_error == 0
                % only the winning competitor is active in this domain
                % split its domain, 
                % domain_existent = True, inters_domain = domain of winning
                domain_existent = true; 
                inters_domain = competitors(idx_best).domain;
            else
                %get intersection of winning competitor and competitor with
                %least prediction error
                domain1 = competitors(idx_best).domain;
                domain2 = competitors(idx_least_error).domain;
                if max(domain2) > max(domain1)
                    inters_domain = [min(domain2) max(domain1)];
                else
                    inters_domain = [min(domain1) max(domain2)];
                end
                
                %check if domain already exists
                domain_existent = false;
                for comp = 1:length(competitors)
                    if inters_domain == competitors(comp).domain
                        domain_existent = true;
                    end
                end        
            end
            
        end
        
        function competitors = eliminate_redundant_competitors(obj, competitors)
            %get list with redundant competitors, eliminate them 
            redundant_comp = zeros(length(competitors)); 
            
            for i = 1:length(competitors)
                for j = 1:length(competitors)
                    if i ~= j
                        %check if xi intersected with xj = xi
                        x_i = competitors(i).domain;
                        x_j = competitors(j).domain;
                        
                        if (x_i(1) >= x_j(1)) && (x_i(2) <= x_j(2))
                            %check thresholds
                            %for finite number of points in domain x_i, number
                            %of points is hyperparameter
                            x = linspace(x_i(1), x_i(2), obj.redundance_check_num_points);
                            diff_value = obj.threshold_v - 0.1;
                            diff_relevance = obj.threshold_r - 0.1;
                            sample_i = 1;
                            while((diff_value < obj.threshold_v) && (diff_relevance < obj.threshold_r) && (sample_i <= length(x)))
                                %calculate value and relevance differences
                                [diff_value, diff_relevance] = calculate_diff(obj, competitors(i), competitors(j), x(sample_i));
                                
                                if sample_i == length(x)
                                    %perform other comparison of threshold and
                                    %value
                                    if (diff_value < obj.threshold_v) && (diff_relevance < obj.threshold_r)
                                        %set in list that competitor i should
                                        %be eliminated
                                        redundant_comp(i) = 1;
                                    end
                                end
                                sample_i = sample_i + 1;
                            end
                        end
                    end
                end
            end
            %eliminate redundant competitors
            for comp = 1:length(redundant_comp)
                if redundant_comp(comp) == 1
                    % competitor comp must be eliminated
                    competitors_new = competitors(1:end-1);
                    competitors_new(comp:end) = competitors(comp+1:end);
                    competitors = competitors_new;
                end
            end
        end
          
        function [diff_value, diff_relevance] = calculate_diff(obj, competitor_i, competitor_j, x)
            y_pred_i = gmm(obj, x, competitor_i);
            y_pred_j = gmm(obj, x, competitor_j);
            diff_value = abs(y_pred_i - y_pred_j) / max(abs(y_pred_i), abs(y_pred_j));
            
            gamma_i = relevance_function(obj, x, y_pred_i, competitor_i);
            gamma_j = relevance_function(obj, x, y_pred_j, competitor_j);
            diff_relevance = abs(gamma_i - gamma_j) / max(abs(gamma_i), abs(gamma_j));
        end
        
        function [x, y] = get_observation(obj, domain)
            % get one swap of observations back or forth [-5,5], sample
            % interval 0.1
            if strcmp(domain, "forward")
                x = linspace(-5, 5, 101)';
            end
            if strcmp(domain, "backward")
                x = linspace(5, -5, 101)';
            end
            y = sin(x);
        end
        
        function Gamma = relevance_function(obj, xt, yt, competitor)
            % Input: 
            %
            % xt                 current sample x value
            % yt                 current sample y value
            % competitor         current competitor
            % num_samples        current number of samples in domain
            %
            % Output:
            % 
            % Gamma              evaluated relevance of competitor
            
            %calculate sample pdf in domain
            weighted_1st_gaussian = competitor.alpha_1 * multivariate_normal(obj, [xt; yt], competitor.mu_1, competitor.sigma_1);
            weighted_2nd_gaussian = competitor.alpha_2 * multivariate_normal(obj, [xt; yt], competitor.mu_2, competitor.sigma_2);
            p_i = weighted_1st_gaussian + weighted_2nd_gaussian;
            
            %calculate sample variance
            [~, sigma_i] = gmm(obj, xt, competitor); %or recursive formulation of equation (5) from paper?
            
            % alpha quantile of chi^2 distribution with (num_samples -1) degrees of
            % freedom where
            
            if competitor.n_i == 1
                deg_freedom = 1;
            else
                deg_freedom = competitor.n_i * p_i * obj.V_i - 1; 
            end
            
            if deg_freedom < 1
                deg_freedom = 1;
            end
                
            alpha_quantile = chi2inv(obj.alpha, deg_freedom); 
            
            Gamma = alpha_quantile/(deg_freedom * sigma_i); 
        end
        
        function degenerate_function(obj)
            
        end
        
        function [y_pred, var_pred] = gmm(obj, xt, competitor) % prediction of y and variance for competitor with gmm_params
            % One Gaussian Mixture Model consisting of two Gaussians for
            % one of the competitors (one competitor will have more than
            % one GMM)
            
            % First Gaussian
            mu_1_x = competitor.mu_1(1:end-1);
            mu_1_y = competitor.mu_1(end);
            sigma_1_xx = competitor.sigma_1(1:end-1,1:end-1);
            sigma_1_xy = competitor.sigma_1(1:end-1, end);
            sigma_1_yx = competitor.sigma_1(end, 1:end-1);
            sigma_1_yy = competitor.sigma_1(end,end);
            
            % Second Gaussian
            mu_2_x = competitor.mu_2(1:end-1);
            mu_2_y = competitor.mu_2(end);
            sigma_2_xx = competitor.sigma_2(1:end-1,1:end-1);
            sigma_2_xy = competitor.sigma_2(1:end-1, end);
            sigma_2_yx = competitor.sigma_2(end, 1:end-1);
            sigma_2_yy = competitor.sigma_2(end,end);
            
            % Weighting the two Gaussians with GMM parameters alpha_1 and
            % alpha_2
            weighted_1st_gaussian = competitor.alpha_1 * multivariate_normal(obj, xt, mu_1_x, sigma_1_xx);
            weighted_2nd_gaussian = competitor.alpha_2 * multivariate_normal(obj, xt, mu_2_x, sigma_2_xx);
            
            % Sum of two weighted Gaussians
            sum_weighted_gaussians = weighted_1st_gaussian + weighted_2nd_gaussian;
            if sum_weighted_gaussians == 0
                beta_1 = 0.5;
                beta_2 = 0.5;
            else                
                beta_1 = weighted_1st_gaussian / sum_weighted_gaussians;
                beta_2 = weighted_2nd_gaussian / sum_weighted_gaussians;
            end
            % Variances of two Gaussians
            var_1_y = sigma_1_yy - (sigma_1_yx / sigma_1_xx) * sigma_1_xy;
            var_2_y = sigma_2_yy - (sigma_2_yx / sigma_2_xx) * sigma_2_xy;
            
            % mean of two Gaussians
            mu_1_y_x = mu_1_y + (sigma_1_yx / sigma_1_xx) * (xt - mu_1_x);
            mu_2_y_x = mu_2_y + (sigma_2_yx / sigma_2_xx) * (xt - mu_2_x);
            
            % mean value of function approximation
            mu_y_x = beta_1 * mu_1_y_x + beta_2 * mu_2_y_x;
            
            % variance of function approximation
            var_y_x = beta_1 * (var_1_y + (mu_1_y_x - mu_y_x)^2) + beta_2 * (var_2_y + (mu_2_y_x - mu_y_x)^2);
            
            % prediction
            y_pred = mu_y_x;
            var_pred = var_y_x;         
        end
        
        function [competitor] = initialize_competitor(obj, domain) %create new competitor with 2 Gaussians in domain
            
            %initialize domain
            competitor.domain = domain;
            
            %initialize parameters of the two Gaussians
            %initialize alpha_1
            competitor.alpha_1 = 0.5;
            
            %initialize alpha_2
            competitor.alpha_2 = 0.5;
            
            %initialize number of samples observed in the domain
            competitor.n_i = 0;
            
            xt1 = 1/3 * (domain(1) + domain(2)); 
            xt2 = 2/3 * (domain(1) + domain(2)); 
            
            %initialize mu_1
            competitor.mu_1 = [xt1; sin(xt1)];
            
            %initialize mu_2
            competitor.mu_2 = [xt2; sin(xt2)];
            
            %initialize sigma_1            
            competitor.sigma_1 = 100*eye(length([xt1; 1])); %not correctly initialized! C=1/sqrt(2*pi)*(obj.w_new/(1-obj.w_new) *... 
            
            %initialize sigma_2
            competitor.sigma_2 = 100*eye(length([xt2; 1]));
            
            %initialize cummulative sums
            competitor.W_1 = 1;
            competitor.W_2 = 1;
            
            competitor.X_1 = competitor.mu_1 / competitor.W_1;
            competitor.X_2 = competitor.mu_2 / competitor.W_2;
            
            competitor.XX_1 = competitor.sigma_1 / competitor.W_1 ...
                - competitor.mu_1 * competitor.mu_1';
            
            competitor.XX_2 = competitor.sigma_2 / competitor.W_2 ...
                - competitor.mu_2 * competitor.mu_2';            
        end
        
        function competitors = generate_competitor_splitting(obj, competitors, idx_best)
            %theory part 1: get dimension along which the variance is
            %largest (irrelevant as domain for sine function only has one
            %dimension)
            
            % split domain of winning competitor in three parts, each of
            % half of its size             
            curr_domain = competitors(idx_best).domain;
            length_domain = curr_domain(2) - curr_domain(1);
            
            domain_1 = [curr_domain(1) (curr_domain(1) + 0.5 * length_domain)];
            domain_2 = [(curr_domain(1) + 0.25 * length_domain) (curr_domain(1) + 0.75 * length_domain)];
            domain_3 = [(curr_domain(1) + 0.5 * length_domain) curr_domain(2)];
            
            %should we also remove old competitor?
            %competitors(idx_best:end) = competitors(idx_best+1:end);
            
            %add the new competitors 
            new_idx = length(competitors) + 1;
            competitors(new_idx) = initialize_competitor(obj, domain_1);
            
            new_idx = new_idx + 1;
            competitors(new_idx) = initialize_competitor(obj, domain_2);
            
            new_idx = new_idx + 1;
            competitors(new_idx) = initialize_competitor(obj, domain_3);  
        end
                
        function competitor = update_gmm_parameters(obj, competitor, xt, yt) %update parameters of competitor with sample xt, yt
            
            weighted_1st_gaussian = competitor.alpha_1 * multivariate_normal(obj, [xt; yt], competitor.mu_1, competitor.sigma_1);
            weighted_2nd_gaussian = competitor.alpha_2 * multivariate_normal(obj, [xt; yt], competitor.mu_2, competitor.sigma_2);
            
            sum_weighted_gaussians = weighted_1st_gaussian + weighted_2nd_gaussian;
            
            % E-step
            % activations of the two Gaussians of the GMM
            w_t_1st_gaussian = weighted_1st_gaussian / sum_weighted_gaussians;
            w_t_2nd_gaussian = weighted_2nd_gaussian / sum_weighted_gaussians;
            
            
            % M-step
            it = competitor.n_i; %????
            lambda = 1 - (1 - obj.forget_a)/ (obj.forget_a * it + obj.forget_b);
            
            %update f-values for online EM 
            competitor.W_1 = lambda ^ w_t_1st_gaussian * competitor.W_1 + (1 - lambda ^ w_t_1st_gaussian) / (1 - lambda) * 1;
            competitor.X_1 = lambda ^ w_t_1st_gaussian * competitor.X_1 + (1 - lambda ^ w_t_1st_gaussian) / (1 - lambda) * [xt; yt];
            competitor.XX_1 = lambda ^ w_t_1st_gaussian * competitor.XX_1 + (1 - lambda ^ w_t_1st_gaussian) / (1 - lambda) * [xt; yt] * [xt; yt]';
            
            competitor.W_2 = lambda ^ w_t_2nd_gaussian * competitor.W_2 + (1 - lambda ^ w_t_2nd_gaussian) / (1 - lambda) * 1;
            competitor.X_2 = lambda ^ w_t_2nd_gaussian * competitor.X_2 + (1 - lambda ^ w_t_2nd_gaussian) / (1 - lambda) * [xt; yt];
            competitor.XX_2 = lambda ^ w_t_2nd_gaussian * competitor.XX_2 + (1 - lambda ^ w_t_2nd_gaussian) / (1 - lambda) * [xt; yt] * [xt; yt]';
            
            sum_Wtj = competitor.W_1 + competitor.W_2;
            
            % Update alpha_1
            competitor.alpha_1 = competitor.W_1 / sum_Wtj;
            
            % Update alpha_2
            competitor.alpha_2 = competitor.W_2 / sum_Wtj;
            
            % Update mu_1
            competitor.mu_1 = competitor.X_1 / competitor.W_1;
            
            % Update mu_2
            competitor.mu_2 = competitor.X_2 / competitor.W_2;
            
            % Update sigma_1
            competitor.sigma_1 = competitor.XX_1 / competitor.W_1 - (competitor.mu_1 * competitor.mu_1');
            
            % Update_sigma_2
            competitor.sigma_2 = competitor.XX_2 / competitor.W_2 - (competitor.mu_2 * competitor.mu_2');  
            
            %increment number of samples observed in the domain
            competitor.n_i = competitor.n_i +1;
            
            % Ensure positive definiteness of covariance matrix sigma_1
            veig=eig(competitor.sigma_1);
            while min(veig)<0.000001
                reg_coef = 2; %0.04
                nn=0; %????
                variance = trace(competitor.sigma_1)/(nn+1);
                variance = max(variance,0.01);
                competitor.sigma_1 = competitor.sigma_1 + reg_coef * variance^2 * eye(size(competitor.sigma_1));
                veig = eig(competitor.sigma_1);
            end
            
            % Ensure positive definiteness of covariance matrix sigma_2
            veig=eig(competitor.sigma_2);
            while min(veig)<0.000001
                reg_coef = 2;
                nn=0; %????
                variance = trace(competitor.sigma_2)/(nn+1);
                variance = max(variance,0.01);
                competitor.sigma_2 = competitor.sigma_2 + reg_coef * variance^2 * eye(size(competitor.sigma_2));
                veig = eig(competitor.sigma_2);
            end
        end
        
        function fx = multivariate_normal(obj, x, mu, sigma)
            % Multivariate normal (Gaussian) distribution
            % Input:
            %   mu      mean vector
            %   sigma   covariance matrix
            
            if det(sigma) ==  0
                disp('Error in multivariate normal: sigma has det 0')
            end
            
            % Ensure positive definiteness of covariance matrix Sigma
            veig=eig(sigma);
            while min(veig)<0.000001
                reg_coef = 0.04;
                nn=0; %????
                variance = trace(sigma)/(nn+1);
                variance = max(variance,0.01);
                sigma = sigma + reg_coef * variance^2 * eye(size(sigma));
                veig = eig(sigma);
            end
            
            p = length(x);
            factor = 1 / (sqrt((2 * pi) ^ p * det(sigma)));
            fx = factor * exp(-0.5 * (x - mu)' * inv(sigma) * (x - mu));
        end
        
        function fx = chi_squared(obj, x, k)
            % Chi^2 distribution 
            % 
            % Input:
            %
            %   k       degrees of freedom (important: k>0)
            %   x       argument
            
            factor = 1 / ( 2^(0.5*k) * factorial(k - 1));
            fx = factor * x^(0.5 * k -1) * exp(- 0.5 * x);
        end
        
        function initializeMsePlot(obj)
            figure
            title('Evolution of MSE')
            xlabel('Episodes')
            ylabel('Mean Squared Error')
            axis([0 200 0 10])
            hold on
        end
        
        function plot_mse(obj, mse, episodes)
            plot(episodes, mse, 'o:', 'Color', 'r');
            hold on
            drawnow
        end        
        
        function plot_funcapprox(obj, y_approx, y_var, y)
            stdv = sqrt(y_var);
            y_upper = y_approx + stdv;
            y_lower = y_approx - stdv;
            
            x = linspace(-5, 5, 101);
            
            fig2 = figure;
            set(fig2,'defaultLegendAutoUpdate','off');
            hold on
            title('Function Approximation')
            xlabel('x')
            ylabel('y')
            axis([-5 5 -1.5 1.5])
            patch([x fliplr(x)], [y_upper' fliplr(y_lower')],'c')
            hold on
            plot(x, y,'Color', 'r', 'Linewidth', 1.5);
            hold on
            plot(x, y_approx,'Color', 'b', 'Linewidth', 1.5)
            hold on
            legend('confidence','y','y_{approx}')
        end
    end
    
end