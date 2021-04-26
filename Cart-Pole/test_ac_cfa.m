clear
close all

%% set hyperparameters
params.episodes = 200; %train & test episodes
params.training_iterations = 200; %training iterations per episode

params.V_i = 0.1; %for relevance function 
params.redundance_num_points = 3; %number of points redundance check is performed on 
params.threshold_v = 0.1; %threshold value difference for redundant competitors
params.threshold_r = 0.1; %threshold relevance difference for redundant competitors
params.i_elim = 50; %after how many samples should the redundance check be performed?
params.threshold_n = 50; %minimum confidence threshold for generating new competitors
params.threshold_error = 0.2; %approximation error threshold for generating new competitors
params.domain_overlap = 2; %overlap of domains of competitors

params.forget_a = 0.001; %parameter a for forgetting factor
params.forget_b = 10; %parameter b for forgetting factor
params.number_actions = 5; %number of actions to explore
params.exploration_noise = 2; %how much exploration noise is added in exploration exploitation strategy

params.number_gaussians = 22; %number of Gaussians per competitor
params.gamma = 0.9; %Discount factor for Bellman equation
params.learning_rate = 0.6; %learning rate in gradient descent of policy CFA
params.learning_rate_decay = 0.02; %learning rate decay per episode
params.min_learning_rate = 0.4; %minimum learning rate

%% train model 
%actor_critic_CFA([2 2 2 2 2], [2 2 2 2], params)

%% visualize cart pole with learnt control actions

% Initial state vector and initial control action
a_max = 10;
a_min = -10;
dt = 0.01; % simulation_interval
action_interval = 0.1;
comp_policy = load('comp_p.mat');
z = [0; 0; 0; 0];

%save all evaluated states and actions in vector
z_opt = zeros(4, 100);
u_opt = zeros(1, 100);
curr_reward = 0;
for i=1:10/action_interval %100times
    % Draw action
    u = evaluate_cfa(comp_policy.comp_policy, z, params);
    if u > 10 
        u = 10;
    elseif u < -10
        u = -10; 
    end
    u_opt(i) = u;
    z_opt(:, i) = z;
    
    %action interval: (same action for 0.1s) 
    for j=1:action_interval/dt %10times
        % Evaluate dynamics
        dz = dyn_cartpole(z, u);

        % Calculate next state
        z = next_state(z, dz, dt);
    end
end

z = [0; 0; 0; 0];
u = 0;
frame = 1;
Frames(frame) = draw_cartpole(z,u);
frame = frame + 1;

%display cart-pole
for i=1:10/action_interval %100times
    % Draw action
    u_input = u_opt(i);
    
    %action interval: (same action for 0.1s) 
    for j=1:action_interval/dt %10times
        %display state every 0.01 seconds
        % Evaluate dynamics
        tic;
        dz = dyn_cartpole(z, u_input);

        % Calculate next state
        z = next_state(z, dz, dt);
        Frames(frame) = draw_cartpole(z, u_input);
        frame = frame + 1;
        pause(dt-toc)
    end
end

% create the video writer 
writerObj = VideoWriter('cartpole.mp4', 'MPEG-4');
writerObj.FrameRate = 10;

% open the video writer
open(writerObj);
for i=1:length(Frames)
    % convert the image to a frame
    frame = Frames(i) ;    
    writeVideo(writerObj, frame);
end
% close the writer object
close(writerObj);
%% plot state trajectories
figure;

% time vector
t=0.1:action_interval:10;

% position
subplot(3,1,1)
plot(t, z_opt(1,:), 'DisplayName', 'AC-CFA');
ylabel('Position in m')
xlabel('Time in s')
legend
grid on;

%text file for position
writematrix([t', z_opt(1,:)'],'CFA_z1.txt', 'Delimiter','tab');

% angle
subplot(3,1,2)
plot(t, z_opt(3,:), 'DisplayName', 'AC-CFA');
ylabel('Angle in rad')
xlabel('Time in s')
legend
grid on;

%rescale the angle to get a comparable plot
for i = 1:100
    if z_opt(3, i) < 0
        z_opt(3, i) = z_opt(3, i) + 2 * pi;
    end
end

% text file for angle
writematrix([t', z_opt(3,:)'],'CFA_z3.txt', 'Delimiter','tab');

% input
subplot(3,1,3)
plot(t, u_opt, 'DisplayName', 'AC-CFA');
hold on;
ylabel('Input Force in N')
xlabel('Time in s')
legend
grid on;

%write text file for force
writematrix([t', u_opt'],'CFA_u.txt', 'Delimiter','tab');

%write text file for reward
reward_over_time = zeros(100, 1);
for i = 1:100
    reward_over_time(i) = reward(z_opt(:, i), u_opt(i));
end
writematrix([t', reward_over_time],'CFA_reward.txt', 'Delimiter','tab')
