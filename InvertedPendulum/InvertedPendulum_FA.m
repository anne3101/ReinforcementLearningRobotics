classdef InvertedPendulum_FA
    
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
    end
    
    methods 
        %% exercise 2
        
        function Q=pendulum_Q_learning(obj)
            Q=initializeQ(obj);
            initializeRewardPlot(obj);
            for i=1:obj.number_episodes
                [Q] = trainPendulum(obj,Q);
                reward = testPendulum(obj, Q); 
                plotReward(obj, reward, i);
                %i
            end
            savefig('rewardPlot.fig');
            visualizeBehavior(obj, Q);
        end
        
        function Q=initializeQ(obj)
            %define split limits
            splits_state1 = linspace(-pi, pi, obj.partition_number(1)+1);
            splits_state2 = linspace(obj.velocity_limits(1), obj.velocity_limits(2), obj.partition_number(2)+1);
            splits_action = linspace(obj.torque_limits(1), obj.torque_limits(2), obj.partition_number(3)+1);
            
            %matrix Q with columns: 
            %  ["lower limit state1" "upper limit state1" "lower limit state2" "upper limit state2" ...
            %     "lower limit action" "upper limit action" "number updates" "mean Q-Value" "variance of Q"]
            %initialize Q-values and number updates with 0, variance with 1
            for i=1:length(splits_state1)-1
                for j=1: length(splits_state2)-1
                    for k=1: length(splits_action)-1
                        if i==1 && j==1 && k==1
                            Q=[splits_state1(1) splits_state1(2) splits_state2(1) splits_state2(2) splits_action(1) splits_action(2) 0 0 2];
                        else
                            Q=[Q; splits_state1(i) splits_state1(i+1) splits_state2(j) splits_state2(j+1) splits_action(k) splits_action(k+1) 0 0 2];
                        end
                    end
                end
            end
        end
        
        function [Q, policy] = trainPendulum(obj,Q)
            state = obj.initial_state;
            for i=1:obj.training_iterations
                [Q, state]=Q_Learning(obj, Q, state); %calculate action and Q-values
            end
        end
        
        function [Q, new_state]=Q_Learning(obj, Q, state) %done
            %select action according to exploitation-exploration strategy
            action = getA_Q_rand(obj, Q, state);
            
            %execute a, get reward r(s,a) and observe new state s'
            new_state = state;
            for j=1:obj.action_interval/obj.differential_time %execute action for 0.1s -> 100 times
                    new_state = get_new_state(obj, new_state, action); %calculate resulting state, keep training with it 
            end 
            [idx_ns] = get_state_index(obj, Q, new_state);
            reward = -abs(state(1));
            
            %estimate maximum Qmax=max(Q(s',a'))
            Q_max = max(Q(idx_ns, 8));
            
            %generate q(s,a)=r(s,a)+gamma*Qmax
            q = reward + obj.discount_factor * Q_max;
            
            %update Q(s,a) using sample (s,a,q(s,a))
            Q = updateQ(obj, Q, q, state, action);
        end
        
        function action = getA_Q_rand(obj, Q, state) %done
            %get number updates, mean Q-values and variance of current state
            [idx_s] = get_state_index(obj, Q, state);
            n_i=[Q(idx_s, 7)]; 
            Q_mean=[Q(idx_s, 8)]; 
            Q_var=[Q(idx_s, 9)]; 
            
            %possible action intervall for current state
            actions=[Q(idx_s, 5:6)];
            
            %get random distributed Q values for each action
            Q_rand = Q_mean + randn(length(Q_mean),1).*Q_var; %trnd(var ./ ni, [obj.partition_number(3), 1]);
            
            %choose action with maximum Q_rand
            [~, action_ind] = max(Q_rand);  
            action = 0.5*(actions(action_ind,1)+actions(action_ind,2));
        end
    
        function Q = updateQ(obj, Q, q, state, action) %done
            [idx_sa] = get_state_action_index(obj, Q, state, action); %Select part i containing sample
            
            %get number updates, mean Q-values and variance of current state and action
            n_i=Q(idx_sa, 7); 
            Q_mean=Q(idx_sa, 8); 
            Q_var=Q(idx_sa, 9); 
            
            %update values
            n_i=n_i+1;
            Q(idx_sa, 7)=n_i;
            
            Q_mean=Q_mean+ obj.alpha * (q - Q_mean); %Qi ←Qi +α(q−Qi)
            Q(idx_sa, 8)=Q_mean;
            
            Q_var= Q_var + obj.alpha * ((q - Q_mean)^2 - Q_var); %Sigma^2 ← Sigma^2 + alpha*((q − Qi)^2 − Sigma^2)
            Q(idx_sa, 9)=Q_var;
            
            if (Q_var >= obj.threshold_var) && (n_i >= obj.threshold_n)
                %splitting critera fulfilled, split cell in half along
                %largest dimension
                [curr_size, split_axis] = max(obj.partition_number);
                obj.partition_number(split_axis) = curr_size +1; 
                Q(idx_sa, 7)=0;
                
                %update current partition by lowering the upper limit
                lowerLim=Q(idx_sa, split_axis*2-1);
                upperLim=Q(idx_sa, split_axis*2);
                upperLim_new=lowerLim+(upperLim-lowerLim)/2;
                Q(idx_sa, split_axis*2)=upperLim_new;
                
                %adding line to Q matrix for the new partition as copy of
                %the old partition
                Q=[Q; Q(idx_sa, :)];
                
                %updating limits of the new partition
                Q(end, split_axis*2-1)=upperLim_new;
                Q(end, split_axis*2)=upperLim;
                
                %set update counter to 0 of new partition
                Q(end, 7)=0;
            end            
        end
                
        function [idx_s] = get_state_index(obj, Q, state)%done
            idx_s = [];
            for i = 1:length(Q)
                if state(1) >= Q(i,1) && (state(1) < Q(i,2) || Q(i,2)==pi)
                    if state(2) >= Q(i,3) && (state(2) < Q(i,4) || Q(i,4)==obj.velocity_limits(2))
                        idx_s = [idx_s; i];
                    end
                end
            end
        end
        
        function [idx_sa] = get_state_action_index(obj, Q, state, action) %done
            idx_sa = [];
            for i = 1:length(Q)
                if state(1) >= Q(i,1) && (state(1) < Q(i,2) || Q(i,2)==pi)
                    if state(2) >= Q(i,3) && (state(2) < Q(i,4) || Q(i,4)==obj.velocity_limits(2))
                        if action >= Q(i,5) && (action < Q(i,6) || Q(i,6)==obj.torque_limits(2))
                            idx_sa = i;
                            return
                        end
                    end
                end
            end
        end
        
        function [reward] = testPendulum(obj, Q)
            reward = 0;
            state = obj.initial_state;
            for k = 1:500
                action = getAction(obj, Q, state); 
                for j=1:obj.action_interval/obj.differential_time %execute action for 0.1s -> 100 times
                    state = get_new_state(obj, state, action); %calculate resulting state, keep training with it 
                end 
                reward = reward + (-abs(state(1)));
            end    
        end
        
        function [action] = getAction(obj, Q, state)
            idx_s = get_state_index(obj, Q, state);
            Q_mean=[Q(idx_s, 8)]; 
            
            %possible action intervall for current state
            actions=[Q(idx_s, 5:6)];
            
            %choose action with maximum Q_mean
            [~, action_ind] = max(Q_mean);  
            action = 0.5*(actions(action_ind,1)+actions(action_ind,2));
        end

        
        function initializeRewardPlot(obj)
            fig = figure;
            set(fig,'defaultLegendAutoUpdate','off');
            title('Evolution of Accumulated Reward')
            hold on
            xlabel('Test episodes') 
            ylabel('Accumlated Reward') 
            axis([0 obj.number_episodes -2000 0])
            
            h = zeros(2, 1);
            h(1) = plot(NaN, NaN, 'o:', 'Color', 'g','DisplayName','Q-learning with FA');
            h(2) = plot(NaN, NaN,'or', 'DisplayName','Q-learning with GMMFA');
            legend(h,'AutoUpdate','off');
            hold on
        end
        
        function plotReward(obj, reward, it)
            plot(it, reward, 'o:', 'Color', 'g');
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
            s(2) = old_s(2) + obj.differential_time * theta_dot_dot; % get new theta_dot 
            s(1) = old_s(1) + obj.differential_time * old_s(2) + 0.5 * obj.differential_time^2 * theta_dot_dot; %get new theta 
            
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
        
        function visualizeBehavior(obj, Q)
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
                a = getAction(obj, Q, s);
                
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
        
    end
    
end