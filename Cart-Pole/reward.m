function r = reward(z, u)
% Calculates the reward for state z and action u
% Input:        z       state vector z = z = [z(1), z(2), z(3), z(4)]'
%                                      = [x_1, dx_1, theta_2, dtheta_2]'
%               u       control action (force)
% Output:       r       reward for state z and action u

% % % % length of pendulum
l = 0.6;
A = 1;

T_inv = A^2 * [1, l, 0; l, l^2, 0; 0, 0, l^2];

% current position of cart-pole
j = [z(1), sin(z(3)), cos(z(3))]; 

% target position of cart-pole
j_target = [0,0,-1]; 

j_diff = j - j_target;

% Reward
r = -(1 - exp(-0.5 * j_diff * T_inv * j_diff'));
end