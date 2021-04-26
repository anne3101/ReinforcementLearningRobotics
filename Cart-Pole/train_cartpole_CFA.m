function [comp_policy, comp_Q] = train_cartpole_CFA(comp_policy, comp_Q, episode, params)
%% performs one training episode
%Inputs: 
%   comp_policy: struct; current competitors that approximate the policy
%   comp_Q: struct; current competitors that approximate the action-value
%   function
%   episode: current training episode
%   params: struct containing hyperparameters
%Outputs:
%   comp_policy: struct; updated competitors that approximate the policy
%   comp_Q struct; updated competitors that approximate the action-value
%   function

%set hyperparameters
gamma = params.gamma;
training_iterations = params.training_iterations;
exploration_noise = params.exploration_noise;
learning_rate = params.learning_rate - episode * params.learning_rate_decay;
if learning_rate < params.min_learning_rate
    learning_rate = params.min_learning_rate;
end
number_actions = params.number_actions;

%% training
state = [0,0,0,0]'; %initial state

for i = 1:training_iterations
    %% select action at randomly for current state
    actions = linspace(-10, 10, number_actions); %linearly spaced actions to evaluate
    Qrnd_max = -inf;
    best_action = 0;
    
    for a = 1:number_actions
        [Qrnd, var] = evaluate_cfa(comp_Q, [state; actions(a)], params);
        Qrnd = Qrnd + normrnd(0, var);
        if Qrnd > Qrnd_max %choose action corresponding to max Qrnd
            Qrnd_max = Qrnd;
            best_action = actions(a);
        end
    end
    
    action = best_action + (rand - 1) * exploration_noise; %add some random noise to action
    
    %ensure that action remains within bounds
    if action > 10
        action = 10;
    elseif action < -10
        action = -10;
    end
    
    %% execute at, get reward,q-value and new state st+1
    
    new_state = state;
    for j= 1:1/0.1 %execute action for 0.1s -> 10 times with simulation interval 0.01
        %evaluate dynamics
        dz = dyn_cartpole(new_state, action);
        
        % calculate new state
        new_state = next_state(new_state, dz, 0.01);
    end
    curr_reward = reward(state, action);
    
    % generate q(st, at) = reward(st, at) + gamma * Q^pi(st+1, at+1)
    [a_t_1, ~] = evaluate_cfa(comp_policy, new_state, params);
    
    %check that action remains within bounds
    if a_t_1 > 10
        a_t_1 = 10;
    elseif a_t_1 < -10
        a_t_1 = -10;
    end
    
    [Q_pi_st_new, ~] = evaluate_cfa(comp_Q, [new_state; a_t_1], params); %get predicted Q-value of new state
    q_st_at = curr_reward + gamma * Q_pi_st_new; 
    
    %% update CFAs
    % update CFA of Q-function using (st, at, q(st,at))
    comp_Q = update_CFA(comp_Q, [state; action], q_st_at, i, params);
    
    % generate atarget
    pi_st = evaluate_cfa(comp_policy, state, params);
    
    %ensure that pi_st remains within bounds
    if pi_st > 10
        pi_st = 10;
    elseif pi_st < -10
        pi_st = -10;
    end
    
    Q_pi_st = evaluate_cfa(comp_Q, [state; pi_st], params);  %get corresponding q-value
    
    if q_st_at > Q_pi_st
        %take exploration action as atarget
        atarget = action;
    else
        %execute one step of gradient ascent
        gradient = gradient_normal([state; pi_st], comp_Q, params);
        atarget = pi_st + learning_rate * gradient;
    end
    
    if atarget > 10
        atarget = 10;
    elseif atarget < -10
        atarget = -10;
    end
    
    %update CFA of policy using sample (st, atarget)
    comp_policy = update_CFA(comp_policy, state, atarget, i, params);
    
    %observe new state
    state = new_state;
end
end