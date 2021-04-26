clc 
clear 
close all

mypendulum = InvertedPendulum;

mypendulum.mass = 1;
mypendulum.pendel_length =  1; 
mypendulum.gravity = 9.8; 
mypendulum.mu = 0.01; 
mypendulum.torque_limits = [-5 5]; 
mypendulum.velocity_limits = [-2*pi, 2*pi];
mypendulum.differential_time =  0.001; 
mypendulum.action_interval = 0.1; 
mypendulum.initial_state = [pi; 0];
mypendulum.display_time = 10;

mypendulum.visualize_time = 0.01; %dt
mypendulum.training_iterations = 500;
mypendulum.discount_factor = 0.9;
mypendulum.alpha = 0.9;
mypendulum.threshold_var = 0.01;
mypendulum.partition_number = [10; 10; 10];
mypendulum.threshold_n = 100;
mypendulum.number_episodes=500;

%% exercise 6 part 1
%task 1:
mypendulum.threshold_mse = 0.15;
mypendulum.a = 0.01;
mypendulum.b = 10;
mypendulum.threshold_error = 0.1;
mypendulum.threshold_density = 0.05;
mypendulum.gmmfa_learning();

%% Q-learning with FA
mypendulum_FA = InvertedPendulum_FA;

mypendulum_FA.mass = 1;
mypendulum_FA.pendel_length =  1; 
mypendulum_FA.gravity = 9.8; 
mypendulum_FA.mu = 0.01; 
mypendulum_FA.torque_limits = [-5 5]; 
mypendulum_FA.velocity_limits = [-2*pi, 2*pi];
mypendulum_FA.differential_time = 0.001; 
mypendulum_FA.action_interval = 0.1; 
mypendulum_FA.initial_state = [pi; 0];
mypendulum_FA.display_time = 10;

mypendulum_FA.visualize_time = 0.01; %dt
mypendulum_FA.training_iterations = 500;
mypendulum_FA.discount_factor = 0.9;
mypendulum_FA.alpha = 0.9;
mypendulum_FA.threshold_var = 0.01;
mypendulum_FA.partition_number = [10; 10; 10];
mypendulum_FA.threshold_n = 100;
mypendulum_FA.number_episodes=500;

Q=pendulum_Q_learning(mypendulum_FA);

%% Exercise 6 task 2:
mypendulum.threshold_mse = 1;
mypendulum.a = 0.001;
mypendulum.b = 10;
mypendulum.discount_factor = 0.99;
mypendulum.threshold_error = 0.3;
mypendulum.threshold_density = 0.01;
gmm_params=pendulum_Q_learning(mypendulum);