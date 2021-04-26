classdef InvertedPendulum
    
    properties 
        mass
        pendel_length
        gravity
        mu 
        torque_limits
        velocity_limits
        differential_time
        action_interval
        initial_state
        display_time
        visualize_time
        training_iterations %number of iterations for trainings epsiode
        number_episodes %number of test episodes before visualization
        discount_factor
        threshold_var %variance threshold for splitting criteria
        alpha %learning rate for updating mean and variance of Q
        partition_number %initial number of partitions per dimension in state-action space 
        threshold_n
        %exercise 6
        threshold_mse
        a %forgetting factor
        b %forgetting factor 
        threshold_error 
        threshold_density
        num_states
    end
    
    methods 
        %% exercise 2
        
        function gmm_params=pendulum_Q_learning(obj)
            %initialize 20 Gaussians
            initial_reward=0;
            for i=1:20
                %draw random torque
                initial_action=(obj.torque_limits(2)-obj.torque_limits(1))*rand(1)+obj.torque_limits(1);
                
                %draw random state
                inititial_position=2*pi*rand(1)-pi;
                inititial_velocity=(obj.velocity_limits(2)-obj.velocity_limits(1))*rand(1)+obj.velocity_limits(1);
                if i==1
                    [gmm_params, old_f_values] = initializeGMM(obj, [inititial_position; inititial_velocity; initial_action; initial_reward]);
                else
                    [gmm_params, old_f_values] = generate_new_gaussian(obj, [inititial_position; inititial_velocity; initial_action], initial_reward, gmm_params, old_f_values);
                end
            end
            
            %initializeRewardPlot(obj);
            openfig('rewardPlot.fig')
            hold on; 
            it = 1;
            for i=1:obj.number_episodes
                [gmm_params, old_f_values, it] = trainPendulum(obj,gmm_params, old_f_values, it);
                reward = testPendulum(obj, gmm_params, old_f_values); 
                plotReward(obj, reward, i);
                i
            end
            visualizeBehavior(obj, gmm_params);
        end
        
        function [gmm_params, old_f_values, it] = trainPendulum(obj,gmm_params, old_f_values, it)
            state = obj.initial_state;
            for i=1:obj.training_iterations
                [gmm_params, old_f_values, state]=Q_Learning(obj, gmm_params, old_f_values, state, it); %calculate action and Q-values
                it = it +1;
            end
        end
        
        function [gmm_params, old_f_values, new_state]=Q_Learning(obj, gmm_params, old_f_values, state, it) 
            %select action according to exploitation-exploration strategy
            action = getA_Q_rand(obj, gmm_params, state);
            
            %execute a, get reward r(s,a) and observe new state s'
            new_state = state;
            for j=1:obj.action_interval/obj.differential_time %execute action for 0.1s -> 100 times
                    new_state = get_new_state(obj, new_state, action); %calculate resulting state, keep training with it 
            end 
            reward = -abs(state(1));
            
            %estimate maximum Qmax=max(Q(s',a'))
            actions = obj.torque_limits(1):0.1:obj.torque_limits(2);
            [q_pred, ~] = estimateQ(obj, gmm_params, state, actions);
            Q_max = max(q_pred);
            
            %generate q(s,a)=r(s,a)+gamma*Qmax
            q = reward + obj.discount_factor * Q_max;
            
            %update Q(s,a) using sample (s,a,q(s,a))
            [old_f_values, gmm_params] = updateGaussians(obj, [state; action], q, gmm_params, old_f_values, it);
        end
        
        function [q_pred, q_var] = estimateQ(obj, gmm_params, state, actions)
            q_pred = zeros(size(actions));
            q_var = zeros(size(actions));
            for i=1:length(actions)
                [q_pred(i), q_var(i)] = predict_y(obj, [state; actions(i)], gmm_params); 
            end
        end
        
        function action = getA_Q_rand(obj, gmm_params, state)
            %get mean Q-values and variance of current state
            actions = obj.torque_limits(1):0.1:obj.torque_limits(2);
            [q_pred, q_var] = estimateQ(obj, gmm_params, state, actions);
            
            %get random distributed Q values for each action
            Q_rand = q_pred + randn(size(q_pred)).*q_var; %trnd(var ./ ni, [obj.partition_number(3), 1]);
            
            %choose action with maximum Q_rand
            [~, action_ind] = max(Q_rand);  
            action = actions(action_ind);
        end
    
        function [reward] = testPendulum(obj, gmm_params, old_f_values)
            reward = 0;
            state = obj.initial_state;
            for k = 1:500
                action = getAction(obj, gmm_params, state); 
                for j=1:obj.action_interval/obj.differential_time %execute action for 0.1s -> 100 times
                    state = get_new_state(obj, state, action); %calculate resulting state, keep training with it 
                end 
                reward = reward + (-abs(state(1)));
            end    
        end
        
        function [action] = getAction(obj, gmm_params, state)
            %get mean Q-values and variance of current state
            actions = obj.torque_limits(1):0.1:obj.torque_limits(2);
            [q_pred, ~] = estimateQ(obj, gmm_params, state, actions);
            
            %choose action with maximum Q_rand
            [~, action_ind] = max(q_pred);  
            action = actions(action_ind);
        end

        function initializeRewardPlot(obj)
            figure
            title('Evolution of Accumulated Reward')
            xlabel('Test episodes') 
            ylabel('Accumlated Reward') 
            axis([0 obj.number_episodes -2000 0])
            hold on
        end
        
        function plotReward(obj, reward, it)
            plot(it, reward, 'o:', 'Color', 'r');
            hold on
            drawnow
        end        
        
        %% exercise 1
        function [pend, ball] = plot_pendulum_action(obj, s, O, pend, ball, clock_diff)
            %delete old elements
            delete(pend);
            delete(ball);
            
            %display position
            P = obj.pendel_length * [-sin(s(1)) cos(s(1))];
            pend = line([O(1) P(1)],[O(2) P(2)]);
            ball = viscircles(P, 0.05);
            pause(obj.visualize_time-(clock-clock_diff)) %wait till visualize_time has passed since clock start
        end
               
        function action = draw_action_uniformly(obj)
            a = obj.torque_limits(1); 
            b = obj.torque_limits(2); 
            action = a + (b-a)*rand;
        end
        
        function s = get_new_state(obj, old_s, action)
            %s(1)... theta
            %s(2)... theta_dot
            
            a = -obj.mu / (obj.mass * obj.pendel_length^2);
            b = obj.gravity / obj.pendel_length;
            c = action / (obj.mass * obj.pendel_length^2);
            
            theta_dot_dot = a * old_s(2) + b * sin(old_s(1)) + c; % get theta_dot_dot
            s(2,1) = old_s(2) + obj.differential_time * theta_dot_dot; % get new theta_dot 
            s(1,1) = old_s(1) + obj.differential_time * old_s(2) + 0.5 * obj.differential_time^2 * theta_dot_dot; %get new theta 
            
            %rescale 
           	while (abs(s(1))) > pi
                if s(1) > pi
                    s(1) = s(1) - 2 * pi;
                elseif s(1) < -pi
                    s(1) = s(1) + 2 * pi;
                end
            end
            
            if s(2) > obj.velocity_limits(2)
                s(2) = obj.velocity_limits(2);
            elseif s(2) < obj.velocity_limits(1)
                s(2) = obj.velocity_limits(1);
            end           
        end
        
        function random_pendulum(obj)
            s = obj.initial_state;
            
            %set up figure 
            figure; 
            hold on;
            O = [0 0]; 
            axis(gca, 'equal');
            axis([-1.1 1.1 -1.1 1.1]);
            grid on; 
            viscircles(O, 0.01);
            P = obj.pendel_length * [-sin(s(1)) cos(s(1))];
            pend = line([O(1) P(1)],[O(2) P(2)]);
            ball = viscircles(P, 0.05);
            
            %displaying time:
            for i=1:obj.display_time/obj.action_interval %100times
                a = draw_action_uniformly(obj);
                
                %action intervall: (same action for 0.1s)
                for j=1:obj.action_interval/obj.differential_time %100times
                    clock_diff=clock;
                    s = get_new_state(obj, s, a);
                    
                    if mod(j,round(obj.visualize_time/obj.differential_time))==0 %plot position every 0.005s
                        [pend, ball] = plot_pendulum_action(obj, s, O, pend, ball, clock_diff);
                    end
                end
            end
        end
        
        function visualizeBehavior(obj, gmm_params)
            s = obj.initial_state;
            
            %set up figure 
            figure; 
            hold on;
            O = [0 0]; 
            axis(gca, 'equal');
            axis([-1.1 1.1 -1.1 1.1]);
            grid on; 
            viscircles(O, 0.01);
            P = obj.pendel_length * [-sin(s(1)) cos(s(1))];
            pend = line([O(1) P(1)],[O(2) P(2)]);
            ball = viscircles(P, 0.05);
            
            %displaying time:
            for i=1:obj.display_time/obj.action_interval %100times
                a = getAction(obj, gmm_params, s);
                
                %action intervall: (same action for 0.1s)
                for j=1:obj.action_interval/obj.differential_time %100times
                    clock_diff=clock;
                    s = get_new_state(obj, s, a);
                    
                    if mod(j,round(obj.visualize_time/obj.differential_time))==0 %plot position every 0.005s
                        [pend, ball] = plot_pendulum_action(obj, s, O, pend, ball, clock_diff);
                    end
                end
            end
            
        end
        
        %% exercise 6: Function Approximation with GMM
        
        function gmmfa_learning(obj)
           mse = obj.threshold_mse + 1;
           initializeMsePlot(obj);
           
           %generate first Gaussian
           [gmm_params, old_f_values] = initializeGMM(obj, [1; 1]);
                    
           episodes = 0;
           it = 0;
           while (mse > obj.threshold_mse && episodes < 1000)
               for domain=["forward", "backward"]
                   %get observation for domain 
                   [x, y] = get_observation(obj, domain); %swap of observations
                   for i=1:length(x)
                       [old_f_values, gmm_params] = updateGaussians(obj, x(i,:), y(i), gmm_params, old_f_values, it);
                       it = it +1;
                   end
                   
                   %get MSE of current domain
                   mse = 0;
                   y_pred = zeros(size(x));
                   y_var = zeros(size(x));
                   
                   for i=1:length(x)
                       %get prediction
                       [y_pred(i), y_var(i)] = predict_y(obj, x(i,:), gmm_params);
                       
                       %get prediction error
                       mse = mse + (y_pred(i) - y(i))^2;
                   end
                   plot_mse(obj, mse, episodes);
                   episodes = episodes + 1;
               end
           end
           %plot last prediction 
           plot_funcapprox(obj, y_pred, y_var, y)
        end
        
        function [old_f_values, gmm_params] = updateGaussians(obj, xt, yt, gmm_params, old_f_values, it)
            %calculate activation of each gaussian
            [old_f_values] = calculate_activation(obj, xt, yt, gmm_params, old_f_values, it);
            
            %update the parameters of GMM
            [gmm_params] = update_gaussians(obj, gmm_params, old_f_values);
            
            %get prediction
            [y_pred, y_var] = predict_y(obj, xt, gmm_params);
            
            %get prediction error
            e_pred = (y_pred - yt)^2;
            
            if e_pred>obj.threshold_error
                %calculate density
                density = get_density(obj, xt, yt, gmm_params);
                
                if density< obj.threshold_density
                    %generate new Gaussian
                    [gmm_params, old_f_values] = generate_new_gaussian(obj, xt, yt, gmm_params, old_f_values);
                end
            end
        end
        
        function [gmm_params, old_f_values] = initializeGMM(obj, obs)
            idx_new_gaussian = 1; 
            
            %initialize parameters of Gaussian
            gmm_params(idx_new_gaussian).mu = [obs];
            gmm_params(idx_new_gaussian).covariance = eye(length(obs)); %not correctly initialized! 
            
            %initialize cummulative sums
            old_f_values(idx_new_gaussian).W = 1;
            old_f_values(idx_new_gaussian).X = gmm_params(idx_new_gaussian).mu * old_f_values(idx_new_gaussian).W;
            old_f_values(idx_new_gaussian).XX = old_f_values(idx_new_gaussian).W * (gmm_params(idx_new_gaussian).covariance ...
                + gmm_params(idx_new_gaussian).mu*gmm_params(idx_new_gaussian).mu');
            
            %initialize alpha
            gmm_params(idx_new_gaussian).alpha = 1;   
        end
        
        function [x, y] = get_observation(obj, domain)
            % get one swap of observations back or forth [-5,5], sample
            % interval 0.1
            if strcmp(domain, "forward")
                x = linspace(-5, 5, 101)';
            else
                x = linspace(5, -5, 101)';
            end
            y = sin(x);
        end
                
        function [gmm_params, old_f_values] = generate_new_gaussian(obj, xt, yt, gmm_params, old_f_values)
            idx_new_gaussian = length(gmm_params) + 1; 
            
            %initialize parameters of Gaussian
            gmm_params(idx_new_gaussian).mu = [xt; yt];
            gmm_params(idx_new_gaussian).covariance = eye(length(gmm_params(idx_new_gaussian).mu)); %not correctly initialized! 
            
            %initialize cummulative sums
            old_f_values(idx_new_gaussian).W = 1;
            old_f_values(idx_new_gaussian).X = gmm_params(idx_new_gaussian).mu * old_f_values(idx_new_gaussian).W;
            old_f_values(idx_new_gaussian).XX = old_f_values(idx_new_gaussian).W * (gmm_params(idx_new_gaussian).covariance ...
                + gmm_params(idx_new_gaussian).mu*gmm_params(idx_new_gaussian).mu');
            
            %initialize alpha
            sum_W_tj = 0; 
            for idx_gaussian = 1:length(gmm_params)
                sum_W_tj = sum_W_tj + old_f_values(idx_gaussian).W;
            end
            if sum_W_tj == 0
                sum_W_tj = 1;
            end
            gmm_params(idx_new_gaussian).alpha = old_f_values(idx_new_gaussian).W / sum_W_tj;            
        end
               
        function [old_f_values] = calculate_activation(obj, xt, yt, gmm_params, old_f_values, it)
            %calculate activation for each Gaussian, E-step
            sum_gmm = 0;
            for i = 1:length(gmm_params) % i: index of corresponding Gaussian
                interm = multivariate_normal(obj, [xt; yt], gmm_params(i).mu, gmm_params(i).covariance);
                sum_gmm = sum_gmm + gmm_params(i).alpha * interm; 
            end
            for i = 1:length(gmm_params)
                %E-Step
                wti = gmm_params(i).alpha * multivariate_normal(obj, [xt; yt], gmm_params(i).mu, gmm_params(i).covariance) / sum_gmm;
                
                %M-step
                lambda = 1 - (1 - obj.a) / (obj.a * it + obj.b); %missing: local adjustment of forgetting factor
                old_f_values(i).W = lambda ^ wti * old_f_values(i).W + (1 - lambda ^ wti) / (1 - lambda) * 1;
                old_f_values(i).X = lambda ^ wti * old_f_values(i).X + (1 - lambda ^ wti) / (1 - lambda) * [xt; yt];
                old_f_values(i).XX = lambda ^ wti * old_f_values(i).XX + (1 - lambda ^ wti) / (1 - lambda) * [xt; yt] * [xt; yt]';
            end
        end
        
        function [gmm_params] = update_gaussians(obj, gmm_params, old_f_values)
            %update params, M-step
            sum_Wtj = 0; 
            for j = 1:length(gmm_params) 
                sum_Wtj = sum_Wtj + old_f_values(j).W;
            end
            for i = 1:length(gmm_params)
                gmm_params(i).alpha = old_f_values(i).W / sum_Wtj;
                gmm_params(i).mu = old_f_values(i).X / old_f_values(i).W;
                gmm_params(i).covariance = old_f_values(i).XX / old_f_values(i).W - (gmm_params(i).mu * gmm_params(i).mu');
            end
        end
        
        function [y_pred, y_var] = predict_y(obj, xt, gmm_params)
            %get mu 
            mu_i = zeros(length(gmm_params));
            variance_i = zeros(length(gmm_params),1);
            beta_numerator = zeros(length(gmm_params),1);
            beta = zeros(length(gmm_params),1);
            
            for i = 1:length(gmm_params)
                mu_ix = gmm_params(i).mu(1:end-1);
                mu_iy = gmm_params(i).mu(end);
                covariance_ixx = gmm_params(i).covariance(1:end-1,1:end-1);
                covariance_ixy = gmm_params(i).covariance(1:end-1,end);
                covariance_iyx = gmm_params(i).covariance(end,1:end-1);
                covariance_iyy = gmm_params(i).covariance(end,end);
                
                mu_i(i) = mu_iy + covariance_iyx / covariance_ixx * (xt - mu_ix);
                variance_i(i) = covariance_iyy - covariance_iyx / covariance_ixx * covariance_ixy;
                
                beta_numerator(i) = gmm_params(i).alpha * multivariate_normal(obj, xt, mu_ix, covariance_ixx);
            end
            
            y_pred = 0; %predicted output
            y_var = 0; %predicted variance of output
            for i = 1:length(gmm_params)
                beta(i) = beta_numerator(i)/sum (beta_numerator);
                y_pred = y_pred + beta(i)*mu_i(i);
                y_var = y_var + beta(i)*(variance_i(i) + (mu_i(i)-y_pred)^2);
            end          
        end
        
        function fx = multivariate_normal(obj, x, mu, sigma)
            if det(sigma) ==  0
                disp('Error in multivariate normal: sigma has det 0')
            end
            
            % Ensure positive definiteness of covariance matrix Sigma
            veig=eig(sigma);
            breg=0;
            while min(veig)<0.000001
                    breg=1;
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
        
        function density = get_density(obj, x, y, gmm_params)
            density = 0; 
            for i = 1:length(gmm_params)
                density = density + gmm_params(i).alpha * multivariate_normal(obj, [x; y], gmm_params(i).mu, gmm_params(i).covariance);
            end
        end
   
        function initializeMsePlot(obj)
            figure
            title('Evolution of MSE')
            xlabel('Episodes') 
            ylabel('Mean Squared Error') 
            axis([0 200 0 1])
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