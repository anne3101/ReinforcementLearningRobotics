close all;
clear all;

Approximator = CFA_with_GMMFA;

Approximator.threshold_mse = 0.01;
Approximator.forget_a = 0.001;
Approximator.forget_b = 10;
Approximator.threshold_error = 0.05;
Approximator.threshold_n = 200; % threshold for minimum number of sample
Approximator.i_elim = 50; % number of sample when to check for redundant competitors
Approximator.initial_num_competitors = 9;
Approximator.V_i = 0.1;
Approximator.alpha = 0.95; % upper bound of confidence interval 
Approximator.redundance_check_num_points = 5; % number of points to check (for the thresholds when eliminating redundant competitors)
Approximator.threshold_v = 0.2; % threshold for competitor elimination
Approximator.threshold_r = 0.2; % threshold for competior elimination
CFA(Approximator);