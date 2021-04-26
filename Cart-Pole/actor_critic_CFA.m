function actor_critic_CFA(ini_comp_Q, ini_comp_policy, params)
    %% main function that calls the training process 
    %Inputs: 
    %   ini_comp_Q: 5-dim vector that specifies initial number of
    %               competitors per dimension (x, d_x, theta,
    %               d_theta, action)
    %   ini_comp_policy: 4-dim vector that specifies initial number of
    %               competitors per dimension (x, d_x, theta,
    %               d_theta)
    %   params:     hyperparameters
    
    %set hyperparameters
    episodes = params.episodes;
    overlap = params.domain_overlap;
    
    %% initialize CFA of  action policy (actor)
    %-> actor: input: state (4-dim), output: policy
    domains_1 = linspace(-6, 6, ini_comp_policy(1) + 1);  %x
    domains_2 = linspace(-10, 10, ini_comp_policy(2) + 1);  %d_x
    domains_3 = linspace(-pi, pi, ini_comp_policy(3) + 1); %theta
    domains_4 = linspace(-10, 10, ini_comp_policy(4) + 1); %d_theta
    
    i = 1;
    for i_x = 1:ini_comp_policy(1)
        domain = zeros(4,2); 
        %first dimension
        if i_x == 1
            domain(1,:) = [(domains_1(i_x)) (domains_1(i_x + 1)+overlap)];
        elseif i_x == ini_comp_policy(1)
            domain(1,:) = [(domains_1(i_x)-overlap) (domains_1(i_x+1))];
        else
            domain(1,:) = [(domains_1(i_x)-overlap) (domains_1(i_x+1)+overlap)];
        end
        
        for i_dx = 1:ini_comp_policy(2)
            %second dimension
            if i_dx == 1
                domain(2,:) = [(domains_2(i_dx)) (domains_2(i_dx + 1)+overlap)];
            elseif i_dx == ini_comp_policy(2)
                domain(2,:) = [(domains_2(i_dx)-overlap) (domains_2(i_dx+1))];
            else
                domain(2,:) = [(domains_2(i_dx)-overlap) (domains_2(i_dx+1)+overlap)];
            end
            
            for i_t = 1:ini_comp_policy(3)
                %third dimension
                if i_t == 1
                    domain(3,:) = [(domains_3(i_t)) (domains_3(i_t + 1)+overlap)];
                elseif i_t == ini_comp_policy(3)
                    domain(3,:) = [(domains_3(i_t)-overlap) (domains_3(i_t+1))];
                else
                    domain(3,:) = [(domains_3(i_t)-overlap) (domains_3(i_t+1)+overlap)];
                end
                
                for i_dt = 1:ini_comp_policy(4)
                    %fourth dimension
                    if i_dt == 1
                        domain(4,:) = [(domains_4(i_dt)) (domains_4(i_dt + 1)+overlap)];
                    elseif i_dt == ini_comp_policy(4)
                        domain(4,:) = [(domains_4(i_dt)-overlap) (domains_4(i_dt+1))];
                    else
                        domain(4,:) = [(domains_4(i_dt)-overlap) (domains_4(i_dt+1)+overlap)];
                    end
                    comp_policy(i) = initialize_competitor(domain, params, 0);
                    i = i + 1;
                end
            end
        end      
    end
   
    %% initialize CFA of action-value-function (critic)
    %-> output: value fct. 

    domains_1 = linspace(-6, 6, ini_comp_Q(1) + 1);  %x
    domains_2 = linspace(-10, 10, ini_comp_Q(2) + 1);  %d_x
    domains_3 = linspace(-pi, pi, ini_comp_Q(3) + 1); %theta
    domains_4 = linspace(-10, 10, ini_comp_Q(4) + 1); %d_theta
    domains_5 = linspace(-10, 10, ini_comp_Q(5) + 1); %policy (applied force)
    
    i = 1;
    for i_x = 1:ini_comp_Q(1)
        domain = zeros(5,2); 
        %first dimension
        if i_x == 1
            domain(1,:) = [(domains_1(i_x)) (domains_1(i_x + 1)+overlap)];
        elseif i_x == ini_comp_Q(1)
            domain(1,:) = [(domains_1(i_x)-overlap) (domains_1(i_x+1))];
        else
            domain(1,:) = [(domains_1(i_x)-overlap) (domains_1(i_x+1)+overlap)];
        end
        
        for i_dx = 1:ini_comp_Q(2)
            %second dimension
            if i_dx == 1
                domain(2,:) = [(domains_2(i_dx)) (domains_2(i_dx + 1)+overlap)];
            elseif i_dx == ini_comp_Q(2)
                domain(2,:) = [(domains_2(i_dx)-overlap) (domains_2(i_dx+1))];
            else
                domain(2,:) = [(domains_2(i_dx)-overlap) (domains_2(i_dx+1)+overlap)];
            end
            
            for i_t = 1:ini_comp_Q(3)
                %third dimension
                if i_t == 1
                    domain(3,:) = [(domains_3(i_t)) (domains_3(i_t + 1)+overlap)];
                elseif i_t == ini_comp_Q(3)
                    domain(3,:) = [(domains_3(i_t)-overlap) (domains_3(i_t+1))];
                else
                    domain(3,:) = [(domains_3(i_t)-overlap) (domains_3(i_t+1)+overlap)];
                end
                
                for i_dt = 1:ini_comp_Q(4)
                    %fourth dimension
                    if i_dt == 1
                        domain(4,:) = [(domains_4(i_dt)) (domains_4(i_dt + 1)+overlap)];
                    elseif i_dt == ini_comp_Q(4)
                        domain(4,:) = [(domains_4(i_dt)-overlap) (domains_4(i_dt+1))];
                    else
                        domain(4,:) = [(domains_4(i_dt)-overlap) (domains_4(i_dt+1)+overlap)];
                    end
                    for i_p = 1:ini_comp_Q(5)
                        %fifth dimension
                        if i_p == 1
                            domain(5,:) = [(domains_5(i_p)) (domains_5(i_p + 1)+overlap)];
                        elseif i_dt == ini_comp_Q(4)
                            domain(5,:) = [(domains_5(i_p)-overlap) (domains_5(i_p+1))];
                        else
                            domain(5,:) = [(domains_5(i_p)-overlap) (domains_5(i_p+1)+overlap)];
                        end
                        comp_Q(i) = initialize_competitor(domain, params, 0);
                        i = i + 1;
                    end
                end
            end
        end      
    end
    
    %% set up reward plot 
    figure;
    title('Evolution of Accumulated Reward')
    hold on
    xlabel('Test episodes')
    ylabel('Accumlated Reward')
    axis([0 episodes -600 0])
    hold on
    best_reward = -200; 
    rewards = zeros(1, episodes);
    
    %% perform actual training
    for i=1:episodes
        [comp_policy, comp_Q] = train_cartpole_CFA(comp_policy, comp_Q, i, params);
        reward = test_cartpole_CFA(comp_policy, params);
        
        %save competitors corresponding to best reward
        if reward > best_reward
            save('comp_policy.mat', 'comp_policy')
            best_reward = reward; 
        end
      
        rewards(i) = reward;
        plot(1:i, rewards(1:i), '-k');
        drawnow
    end  
end
