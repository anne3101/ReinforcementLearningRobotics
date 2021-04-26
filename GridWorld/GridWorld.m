classdef GridWorld
    % GridWorld
    % 
    
    properties
        num_actions
        num_rows
        num_columns
        num_states
        grid_states
        discountFactor
        threshold
        rewards
        actions
        epsilon
        alpha
        threshold_q
    end
    
    methods
        function state_transition_matrix = get_transition_matrix(obj, action)
            % action up
            if action == 1
                state_transition_matrix = ...
                    [0.9, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0;
                     0.8, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                     0, 0.8, 0.1, 0, 0, 0.1, 0, 0, 0, 0, 0, 0;
                     0.1, 0, 0, 0.8, 0, 0, 0.1, 0, 0, 0, 0, 0;
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                     0, 0, 0.1, 0, 0, 0.8, 0, 0, 0.1, 0, 0, 0;
                     0, 0, 0, 0.1, 0, 0, 0.8, 0, 0, 0.1, 0, 0;
                     0, 0, 0, 0, 0, 0, 0.8, 0.1, 0, 0, 0.1, 0;
                     0, 0, 0, 0, 0, 0.1, 0, 0.8, 0, 0, 0, 0.1;
                     0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0.9, 0, 0;
                     0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0.8, 0.1, 0;
                     0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0.8, 0.1];
            end
            
            % action down
            if action == 2
                state_transition_matrix = ...
                    [0.1, 0.8, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0;
                     0, 0.2, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                     0, 0, 0.9, 0, 0, 0.1, 0, 0, 0, 0, 0, 0;
                     0.1, 0, 0, 0.8, 0, 0, 0.1, 0, 0, 0, 0, 0;
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                     0, 0, 0.1, 0, 0, 0.8, 0, 0, 0.1, 0, 0, 0;
                     0, 0, 0, 0.1, 0, 0, 0, 0.8, 0, 0.1, 0, 0;
                     0, 0, 0, 0, 0, 0, 0, 0.1, 0.8, 0, 0.1, 0;
                     0, 0, 0, 0, 0, 0.1, 0, 0, 0.8, 0, 0, 0.1;
                     0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0.1, 0.8, 0;
                     0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0.1, 0.8;
                     0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0.9];
            end
            
            % action left
            if action == 3
                state_transition_matrix = ...
                    [0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                     0.1, 0.8, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                     0, 0.1, 0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                     0.8, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0;
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                     0, 0, 0.8, 0, 0, 0.2, 0, 0, 0, 0, 0, 0;
                     0, 0, 0, 0.8, 0, 0, 0.1, 0.1, 0, 0, 0, 0;
                     0, 0, 0, 0, 0, 0, 0.1, 0.8, 0.1, 0, 0, 0;
                     0, 0, 0, 0, 0, 0.8, 0, 0.1, 0.1, 0, 0, 0;
                     0, 0, 0, 0, 0, 0, 0.8, 0, 0, 0.1, 0.1, 0;
                     0, 0, 0, 0, 0, 0, 0, 0.8, 0, 0.1, 0, 0.1;
                     0, 0, 0, 0, 0, 0, 0, 0, 0.8, 0, 0.1, 0.1];
            end
            
            % action right
            if action == 4
                state_transition_matrix = ...
                    [0.1, 0.1, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0;
                     0.1, 0.8, 0.1, 0, 0, 0, 0,	0, 0, 0, 0, 0;
                     0, 0.1, 0.1, 0, 0,	0.8, 0,	0, 0, 0, 0,	0;
                     0, 0, 0, 0.2, 0, 0, 0.8, 0, 0, 0, 0, 0;
                     0,	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
                     0,	0, 0, 0, 0, 0.2, 0, 0, 0.8, 0, 0, 0;
                     0,	0, 0, 0, 0, 0, 0.1, 0.1, 0,	0.8, 0, 0;
                     0,	0, 0, 0, 0, 0, 0.1, 0, 0.1, 0, 0.8, 0;
                     0,	0, 0, 0, 0, 0, 0, 0.1, 0.1, 0, 0, 0.8;
                     0,	0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0.1, 0;
                     0,	0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.8, 0.1;
                     0,	0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.9];
            end
            
            
        end
        
        function snext_possible = get_possible_snext(obj, state)
            % Given: state              current state
            % Return: snext_possible    set of next possible states given
            %                           the current state
            
                if state == 1
                    snext_possible = [1, 2, 4];        
                end
                
                if state == 2
                    snext_possible = [1, 2, 3];
                end
                
                if state == 3
                    snext_possible = [2, 3, 6];
                end
                    
                if state == 4
                    snext_possible = [1, 4, 7];
                end
                
                if state == 6
                    snext_possible = [3, 6, 9];
                end
                
                if state == 7
                    snext_possible = [4, 7, 8, 10];
                end
                
                if state == 8
                    snext_possible = [7, 8, 9, 11];
                end
                
                if state == 9
                    snext_possible = [6, 8, 9, 12];
                end
                
                if state == 10
                    snext_possible = [7, 10, 11];
                end
                
                if state == 11
                    snext_possible = [8, 10, 11, 12];
                end
                
                if state == 12
                    snext_possible = [9, 11, 12];
                end    
        end
        
        function snext_prob = snext_probabilities(obj, state, snext_possible, action)
            % Given: state          current state
            %        snext_possible set of next possible states given
            %                       current state
            % Return: snext_prob    probabilities of states in set
            %                       snext_possible
            state_transition_matrix = get_transition_matrix(obj, action);
            snext_prob = zeros(size(snext_possible));
            snext_prob_index = 1;
            
            % get probabilities of next states
            for snext_poss = snext_possible
                snext_prob(1, snext_prob_index) = state_transition_matrix(state, snext_poss);
                snext_prob_index = snext_prob_index + 1;
            end
        end
        
        function [snext, reward] = simulator(obj, state, action)
            % Given: state,     in which you are currently
            %        action,    that is chosen
            % Return: snext,    next state generated by environment
            %        r,         reward
            
            % action up
            if action == 1
                
                % get possible next states when being in current state
                snext_possible = get_possible_snext(obj, state);
                
                % get probabilities of set of next states
                snext_prob = snext_probabilities(obj, state, snext_possible, action);
                
                % randomly selext the next state from the set of next
                % states given the probabilities
                snext = randsample(snext_possible,1,true,snext_prob);
                reward = obj.rewards(snext);
                
            end
            
            % action down
            if action == 2
                
                % get possible next states when being in current state
                snext_possible = get_possible_snext(obj, state);
                
                % get probabilities of set of next states
                snext_prob = snext_probabilities(obj, state, snext_possible, action);
                
                % randomly selext the next state from the set of next
                % states given the probabilities
                snext = randsample(snext_possible,1,true,snext_prob);
                reward = obj.rewards(snext);
                
            end
            
            % action left
            if action == 3
                
                % get possible next states when being in current state
                snext_possible = get_possible_snext(obj, state);
                
                % get probabilities of set of next states
                snext_prob = snext_probabilities(obj, state, snext_possible, action);
                
                % randomly selext the next state from the set of next
                % states given the probabilities
                snext = randsample(snext_possible,1,true,snext_prob);
                reward = obj.rewards(snext);
                
            end
            
            % action right
            if action == 4
                
                % get possible next states when being in current state
                snext_possible = get_possible_snext(obj, state);
                
                % get probabilities of set of next states
                snext_prob = snext_probabilities(obj, state, snext_possible, action);
                
                % randomly selext the next state from the set of next
                % states given the probabilities
                snext = randsample(snext_possible,1,true,snext_prob);
                reward = obj.rewards(snext);
                
            end
        end
        
        %% exercise 2
        function [qvalues_final, policy_final, iterations] = sarsa(obj)
            a=0.1; b=1; %parameters for adapting leraning rate
            
            qvalues=rand(12,4); %initialize q
            qvalues(12,:)=zeros(1,4);
            delta=inf;
            iterations=1;
            while (delta > obj.threshold_q || delta == 0) %for each episode
                steps_per_episode=1;
                eps=1;
                previous_qvalues=qvalues;
                
                state=randsample([1:4 6:12],1); %randomly choose start state for each episode
                action=determine_action_eps_greedy(obj, qvalues(state,:), eps); %determine action with epsilon-greedy policy
                while steps_per_episode<100000
                    eps=1/(iterations)^(1/3); %epsilon
                    alpha=1/(a*iterations+b); %learning rate
                    
                    [state_next, reward] = simulator(obj, state, action); %simulate transition
                    action_next=determine_action_eps_greedy(obj, qvalues(state_next,:), eps); %determine action with epsilon-greedy policy
                    qvalues(state, action)=qvalues(state, action)+...
                        alpha*(reward+obj.discountFactor*qvalues(state_next, action_next)-qvalues(state, action)); %update qvalues
                    action=action_next;
                    state=state_next;
                    iterations=iterations+1;
                    steps_per_episode=steps_per_episode+1;
                end
                delta = norm(qvalues - previous_qvalues, Inf);
            end
            
            disp('Using the SARSA algorithm, the Q-values for each action are:')
            disp('Action 1:')
            disp(reshape(qvalues(:,1), [3,4]))
            disp('Action 2:')
            disp(reshape(qvalues(:,2), [3,4]))
            disp('Action 3:')
            disp(reshape(qvalues(:,3), [3,4]))
            disp('Action 4:')
            disp(reshape(qvalues(:,4), [3,4]))
            
            [qvalues_max, policies] = max(qvalues, [], 2);
            qvalues_final = reshape(qvalues_max, [3,4]);
            policy_final = reshape(policies, [3,4]);
            
            disp('The maximum Q-values for each state are:')
            disp(qvalues_final)
            disp('Using the SARSA algorithm, the actions from the final policy are:')
            policy_final(2,2) = 0;
            disp(policy_final)
        end
        
%         function [qvalues_final, policy_final, iterations] = sarsa(obj)
%             
%             qvalues=rand(12,4); %initialize q
%             qvalues(12,:)=zeros(1,4);
%             delta=inf;
%             iterations=1;
%             
%             previous_qvalues=qvalues;
%             state=3;%randsample([1:4 6:12],1); %randomly choose start state for each episode
%             action=determine_action_eps_greedy(obj, qvalues(state,:), obj.epsilon); %determine action with epsilon-greedy policy
%             while (delta > obj.threshold_q || delta == 0)%while steps_per_episode<100000
%                 [state_next, reward] = simulator(obj, state, action); %simulate transition
%                 action_next=determine_action_eps_greedy(obj, qvalues(state_next,:), obj.epsilon); %determine action with epsilon-greedy policy
%                 qvalues(state, action)=qvalues(state, action)+...
%                     obj.alpha*(reward+obj.discountFactor*qvalues(state_next, action_next)-qvalues(state, action)); %update qvalues
%                 action=action_next;
%                 state=state_next;
%                 iterations=iterations+1;
%                 delta = norm(qvalues - previous_qvalues, Inf);
%             end
%             
%             disp('Using the SARSA algorithm, the Q-values for each action are:')
%             disp('Action 1:')
%             disp(reshape(qvalues(:,1), [3,4]))
%             disp('Action 2:')
%             disp(reshape(qvalues(:,2), [3,4]))
%             disp('Action 3:')
%             disp(reshape(qvalues(:,3), [3,4]))
%             disp('Action 4:')
%             disp(reshape(qvalues(:,4), [3,4]))
%             
%             [qvalues_max, policies] = max(qvalues, [], 2);
%             qvalues_final = reshape(qvalues_max, [3,4]);
%             policy_final = reshape(policies, [3,4]);
%             
%             disp('The maximum Q-values for each state are:')
%             disp(qvalues_final)
%             disp('Using the SARSA algorithm, the actions from the final policy are:')
%             policy_final(2,2) = 0;
%             disp(policy_final)
%         end
        
        function action=determine_action_eps_greedy(obj, qvalues_state, eps)
            % Given: obj           current state
            %        qvalues_state Q-values for current state for each action
            %        eps           epsilon for epsilon greedx-policy
            % Return: action    chosen action
            
            [~,action_opt]=max(qvalues_state, [], 2); %determine optimal action
            prob=eps/4*ones(1,4); %probability for each action
            prob(action_opt)=1-eps+eps/4; %probability for optimal action
            action=randsample([1:4],1, true, prob); %choose action with eps-greedy policy
        end
        
        function [qvalues_final, policy_final] = qlearning(obj)
            qvalues = zeros(12, 4);
            s = 3; 
            delta = obj.threshold_q + 1;
            while (delta > obj.threshold_q || delta == 0)
                previous_q = qvalues;
                a = determine_action_eps_greedy(obj, qvalues(s, :), obj.epsilon);
                [snext, r] = simulator(obj, s, a); 
                [Qmax, ~] = max(qvalues(snext,:), [], 2);
                q = r + obj.discountFactor * Qmax;
                qvalues(s, a) = qvalues(s, a) + obj.alpha * (q - qvalues(s,a)); 
                s = snext;
                delta = norm(qvalues - previous_q, Inf);
            end
            [qvalues_max, policies] = max(qvalues, [], 2);
            qvalues_final = reshape(qvalues_max, [3,4]);
            policy_final = reshape(policies, [3,4]);   
            
            disp('The maximum Q-values using q-learning for each state are:')
            disp(qvalues_final)
            disp('Using q-learning, the actions from the final policy are:')
            policy_final(2,2) = 0; 
            disp(policy_final)
        end
        
        %% exercise 1
        function policy = get_random_policy(obj)
            % TO DO no action for state 5
            policy = randi([1,obj.num_actions],12,1);
        end
        
        function values = policyEvaluation(obj, values_init, policy)
            % Input: values_init --> vector containing the values of
            % each state, policy --> current policy
            % Returns: new values of states, delta
            values = values_init;
            delta = obj.threshold + 1;
            
            while (delta > obj.threshold)
                previous_values = values;
                for state = 1:obj.num_states
                    state_transition = get_transition_matrix(obj, policy(state));
                    sum_states = 0;
                    for state_prime = 1:obj.num_states
                        sum_states = sum_states + state_transition(state, state_prime) * previous_values(state_prime);
                    end
                    values(state) = obj.discountFactor * sum_states + obj.rewards(state);
                end
                delta = norm(values - previous_values, Inf);
            end
        end
            
                    
        function policy = policyImprovement(obj, values)
            % Input: values of states of Policy Iteration and corresponding
            %        policy
            % Returns: improved policy
            policies = zeros(12,4);
            for j = 1:obj.num_actions %iterate over actions
                trans = get_transition_matrix(obj, j);
                for state = 1:obj.num_states
                    term = 0; 
                    for state_prime = 1:obj.num_states
                        term = term + trans(state, state_prime) * values(state_prime);
                    end
                    policies(state, j) = obj.discountFactor * term + obj.rewards(state);
                end
            end                 
            % get argmax of policies for each state
            [~, policy] = max(policies, [], 2);
        end
        
        
        function [values,policy] = policyIteration(obj, value_init, init_policy)
            % Iterates through policy Evaluation and policy Improvement
            % until delta is smaller than obj.thresholds
            
            values = value_init;
            delta_policy = obj.threshold + 1;
            policy = init_policy;
            
            while (delta_policy > obj.threshold) 
                previous_policy = policy;
                values = policyEvaluation(obj, values, policy);
                policy = policyImprovement(obj, values);
                delta_policy = norm(policy - previous_policy, Inf);
            end
            
            values = reshape(values, 3,4);
            policy = reshape(policy, 3,4);
            policy(2,2) = 0;        
            
            disp('Using policy iteration, the maximum values are:')
            disp(values)
            disp('Using policy iteration, the actions from the final policy are:')
            disp(policy)
            
        end
        
        function [qvalues_final, policy_final] = ActionValueIteration(obj, qvalues)
            % Q-values size: 12 x 4 (12 states for every action)
            delta_q = obj.threshold + 1;
            while(delta_q > obj.threshold)
                previous_q = qvalues;
                for action = 1:obj.num_actions
                    state_trans = get_transition_matrix(obj, action);
                    for state = 1:obj.num_states
                        sum_states = 0;
                        for state_prime = 1:obj.num_states
                            sum_states = sum_states + state_trans(state, state_prime) * max(previous_q(state_prime,:));
                        end
                        qvalues(state,action) = obj.discountFactor * sum_states + obj.rewards(state);
                    end
                end
                delta_q = norm(qvalues - previous_q, Inf);
            end
            disp('Using action value iteration, the Q-values for each action are:')
            disp('Action 1:')
            disp(reshape(qvalues(:,1), [3,4]))
            disp('Action 2:')
            disp(reshape(qvalues(:,2), [3,4]))
            disp('Action 3:')
            disp(reshape(qvalues(:,3), [3,4]))
            disp('Action 4:')
            disp(reshape(qvalues(:,4), [3,4]))
            
            [qvalues_max, policies] = max(qvalues, [], 2);
            qvalues_final = reshape(qvalues_max, [3,4]);
            policy_final = reshape(policies, [3,4]);
            
            disp('The maximum Q-values for each state are:')
            disp(qvalues_final)
            disp('Using action value iteration, the actions from the final policy are:')
            policy_final(2,2) = 0; 
            disp(policy_final)
        end
    end       
        
end