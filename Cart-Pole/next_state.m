function z = next_state(z, dz, dt)
% Calculates the next state vector z for cart-pole by approximating the ODE
% using Euler's method
%
% Input:
%           z       state vector z = z = [z(1), z(2), z(3), z(4)]'
%                                      = [x_1, dx_1, theta_2, dtheta_2]' 
%           dz      derivatives of state vector dz = [dz(1), dz(2), dz(3), dz(4)]'
%                                                  = [dx_1, ddx_1, dtheta_2, ddtheta_2]'
%           dt      differential time
% Output:   z       state vector z = z = [z(1), z(2), z(3), z(4)]'
%                                      = [x_1, dx_1, theta_2, dtheta_2]'
%           

% New state vector using Euler's method
z = z + dt * dz;

% Adjust to ranges of state vector elements if necessary
% z(1) = [-6,6]
if z(1) > 6
    z(1) = 6;
end

if z(1) < -6
    z(1) = -6;
end

% z(2) = [-10,10]
if z(2) > 10
    z(2) = 10;
end

if z < -10
    z(2) = -10;
end

% z(3) = [-pi, pi]
if z(3) > pi
    z(3) = z(3) - 2 *pi;
end

if z(3) < -pi
    z(3) = z(3) + 2 * pi;
end

% z(4) = [-10, 10]
if z(4) > 10
    z(4) = 10;
end

if z(4) < -10
    z(4) = -10;
end

end