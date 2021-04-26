function curr_reward = test_cartpole_CFA(comp_policy, params)
%executes one test episode
%Inputs:
%   comp_policy: struct of competitors that approximate the policy 
%   params: struct containing the hyperparameters
%Output:
%   curr_reward: reward generated in test episode

curr_reward = 0;
state = [0,0,0,0]'; %start in initial state
dt = 0.01; %simulation_interval

for k = 1:500 %execute 500 actions and get cumulated reward
    %get action by evaluating CFA for policy
    [action, ~] = evaluate_cfa(comp_policy, state, params);
    
    %check that action remains within bounds
    if action > 10
        action = 10;
    elseif action < -10
        action = -10;
    end
    
    for j= 1:1/0.1 %execute action for 0.1s -> 10 times with simulation interval 0.01
        %evaluate dynamics
        dz = dyn_cartpole(state, action);
        
        % calculate new state
        state = next_state(state, dz, dt);
    end
    curr_reward = curr_reward + reward(state, action); %sum up obtained reward
end
end