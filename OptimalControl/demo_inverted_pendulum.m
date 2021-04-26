clc 
clear 
close all

full_DDP = false;

% control limits
Op.lims = [-5 5];

%function
T       = 500;             % horizon
x0      = [pi; 0];       % initial state
u0      = .1*randn(1,T);     % initial control, random
Op.plot = -1;

% optimization problem
DYNCST  = @(x,u,i) pend_dyn_cst(x,u,full_DDP); %need to define dynamics

% run the optimization
[x, u, L, Vx, Vxx, cost, trace, stop] = iLQG(DYNCST, x0, u0, Op);

save('optimal_trajectory.mat', 'x'); 
save('final_action_sequence.mat', 'u');
visualize_final_behavior();

function [f,c,fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu] = pend_dyn_cst(x,u,full_DDP)
u(isnan(u)) = 0;
if nargout == 2
    f = pend_dynamics(x,u); %new_state, from inverted pendulum
    c = pend_cost(x,u);
else
    % state and control indices
    ix = 1:2;
    iu = 3;
    
    % dynamics first derivatives
    xu_dyn  = @(xu) pend_dynamics(xu(ix,:),xu(iu,:));
    J       = finite_difference(xu_dyn, [x; u]);
    fx      = J(:,ix,:);
    fu      = J(:,iu,:);
    
    % second derivatives not calculated as full_DPP = false
    [fxx,fxu,fuu] = deal([]);
    
    % cost first derivatives
    xu_cost = @(xu) pend_cost(xu(ix,:),xu(iu,:));
    J       = squeeze(finite_difference(xu_cost, [x; u]));
    cx      = J(ix,:);
    cu      = J(iu,:);
    
    % cost second derivatives
    xu_Jcst = @(xu) squeeze(finite_difference(xu_cost, xu));
    JJ      = finite_difference(xu_Jcst, [x; u]);
    JJ      = 0.5*(JJ + permute(JJ,[2 1 3])); %symmetrize
    cxx     = JJ(ix,ix,:);
    cxu     = JJ(ix,iu,:);
    cuu     = JJ(iu,iu,:);
    
    [f,c] = deal([]);
end
            end

function s = pend_dynamics(x,u)

a = -0.01;
b = 9.8;
c = u;
s = zeros(size(x));
theta_dot_dot = a * x(2) + b * sin(x(1)) + c; % get theta_dot_dot
s(2,:) = x(2,:) + 0.01 * theta_dot_dot; % get new theta_dot
s(1,:) = x(1,:) + 0.01 * x(2) + 0.5 * 0.01^2 * theta_dot_dot; %get new theta

%rescale
while (abs(s(1))) > pi
    if s(1) > pi
        s(1) = s(1) - 2 * pi;
    elseif s(1) < -pi
        s(1) = s(1) + 2 * pi;
    end
end

if s(2) > pi
    s(2) = pi;
elseif s(2) < -pi
    s(2) = -pi;
end
end

function c = pend_cost(x, u)
c = abs(x(1,:));
end

function y = sabs(x,p)
% smooth absolute-value function (a.k.a pseudo-Huber)
y = pp( sqrt(pp(x.^2,p.^2)), -p);
end

function J = finite_difference(fun, x, h)
% simple finite-difference derivatives
% assumes the function fun() is vectorized

if nargin < 3
    h = 2^-17;
end

[n, K]  = size(x);
H       = [zeros(n,1) h*eye(n)];
H       = permute(H, [1 3 2]);
X       = pp(x, H);
X       = reshape(X, n, K*(n+1));
Y       = fun(X);
m       = numel(Y)/(K*(n+1));
Y       = reshape(Y, m, K, n+1); 
J       = pp(Y(:,:,2:end), -Y(:,:,1)) / h;
J       = permute(J, [1 3 2]);
end

% utility functions: singleton-expanded addition and multiplication
function c = pp(a,b)
c = bsxfun(@plus,a,b);
end

function c = tt(a,b)
c = bsxfun(@times,a,b);
end

function visualize_final_behavior()
x = load('optimal_trajectory.mat');
u = load('final_action_sequence.mat');
s = x.x(:,1);

%set up figure
figure;
hold on;
O = [0 0];
axis(gca, 'equal');
axis([-1.1 1.1 -1.1 1.1]);
grid on;
viscircles(O, 0.01);
P = [-sin(s(1)) cos(s(1))];
pend = line([O(1) P(1)],[O(2) P(2)]);
ball = viscircles(P, 0.05);
sz = size(x.x)
for j=1:sz(2)
    s = x.x(:,j);
    [pend, ball] = plot_pendulum_action(s, O, pend, ball);
    pause(0.005)
end

end

function [pend, ball] = plot_pendulum_action(s, O, pend, ball)
%delete old elements
delete(pend);
delete(ball);

%display position
P = [-sin(s(1)) cos(s(1))];
pend = line([O(1) P(1)],[O(2) P(2)]);
ball = viscircles(P, 0.05);
end
