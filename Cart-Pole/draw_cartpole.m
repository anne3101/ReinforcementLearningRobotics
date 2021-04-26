function Frame = draw_cartpole(z,u)
% Depicts the behavior of the cart-pole with
% Input:        
%           z       state vector z = z = [z(1), z(2), z(3), z(4)]'
%                                      = [x_1, dx_1, theta_2, dtheta_2]'
%           u       control force

% Model parameters of cart-pole
l = 0.6;
x_min = -6;
x_max = 6;
height = 0.2;
width = 0.3;

% Position of cart
cart = [ z(1) + width, height
         z(1) + width, -height
         z(1) - width, -height
         z(1) - width, height
         z(1) + width, height];

% Position of pendulum
pendulum = [z(1), 0; z(1) + 2 * l * sin(z(3)), -2 * l * cos(z(3))];

% Position of wheels
wheels = [z(1)- width, -height
          z(1)+ width, -height];

% Clear current figure window
clf
hold on;

% Plot rail
plot([x_min, x_max], [-height - 0.1, -height - 0.1], 'k', 'linewidth', 2)

% Plot cart
fill(cart(:,1), cart(:,2), 'k', 'edgecolor', 'k');

% Plot wheels
plot(wheels(:,1), wheels(:,2), 'o', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k', 'MarkerSize',10)

% Plot pendulum
plot(pendulum(:, 1), pendulum(:,2), 'linewidth', 2, 'Color', 'r')

axis equal
xlim([-8,8])
ylim([-4,4])
title('Cart-Pole')

Frame = getframe;
drawnow 
end