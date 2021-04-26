function dz = dyn_cartpole(z, u)
% dynamics of cart pole
% Input:
%           z       state vector z = z = [z(1), z(2), z(3), z(4)]'
%                                      = [x_1, dx_1, theta_2, dtheta_2]'
%           u       control force
%           t       time interval
% Output:
%           dz      derivatives of state vector dz = [dz(1), dz(2), dz(3), dz(4)]'
%                                                  = [dx_1, ddx_1, dtheta_2, ddtheta_2]'

% Model parameters of cart-pole
m_1 = 0.5;
m_2 = 0.5;
l = 0.6;
b = 0.1;
g = 9.82;

% intermediate stored variables
m = m_1 + m_2;
s_z_3 = sin(z(3));
c_z_3 = cos(z(3));

% Equations of motions as four decoupled ordinary diff equations
dz_1 = z(2);
dz_2 = (2 * m_2 * l * z(4)^2 * s_z_3 + 3 * m_2 * g * s_z_3 *c_z_3 + 4 * u - 4 * b * z(2))...
    /(4 * m - 3 * m_2 * c_z_3^2);
dz_3 = z(4);
dz_4 = (-3 * m_2 * l * z(4)^2 * s_z_3 * c_z_3 - 6 * m * g * s_z_3 - 6*(u - b * z(2)) * c_z_3)...
    /(4 * l * m - 3 * m_2 * l * c_z_3^2);


dz = [dz_1, dz_2, dz_3, dz_4]';

end