clc
clear all

% Create instance of class Grid_World
gridWorld = GridWorld;

% Set properties of Grid_World instance
gridWorld.num_actions = 4;
gridWorld.num_rows = 3;
gridWorld.num_columns = 4;
gridWorld.num_states = gridWorld.num_rows * gridWorld.num_columns;
gridWorld.discountFactor = 0.9;
gridWorld.threshold = 0.01;
gridWorld.rewards = [0,0,0,0,0,0,0,0,0,1,-100,0]';
gridWorld.actions = 1 : gridWorld.num_actions;
gridWorld.epsilon = 0.05;
gridWorld.alpha = 0.7;

%% Policy Iteration (Exercsise 1)
policy = get_random_policy(gridWorld);
 
[values, new_policy] = policyIteration(gridWorld, zeros(12,1), policy);

%% Action-Value Iteration (Exercise 1)
[qvalues, q_policy] = ActionValueIteration(gridWorld, zeros(12,1));

%% Q-Learning (Exercise 2)
gridWorld.threshold_q = 0.00001;
[qvalues_qlearning, actions_qlearning] = qlearning(gridWorld);

%% Sarsa (Exercise 2)
gridWorld.threshold_q = 0.001;
[qvalues_final, policy_final, iterations] = sarsa(gridWorld);
