function competitor = update_gmm_parameters(competitor, xt, yt, params)
%% updates parameters of competitor with sample xt, yt using online EM
%Inputs: 
%   competitor: competitor whose parameters should be updated
%   (xt, yt): sample used as updating reference 
%   params: struct containing hyperparameters
%Output: 
%   competitor: competitor with updated parameters

%set hyperparameters
forget_a = params.forget_a;
forget_b = params.forget_b;
num_g = params.number_gaussians;

%% E-step
sum_gmm = 0;
for i = 1:num_g % i: index of corresponding Gaussian
    interm = multivariate_normal([xt; yt], competitor.mu(:, i), competitor.sigma(:, :, i));
    sum_gmm = sum_gmm + competitor.alpha(i) * interm;
end

if sum_gmm == 0 || isnan(sum_gmm) %prevent dividing by 0
    sum_gmm = 1; 
end

for i = 1:num_g
    wti = competitor.alpha(i) * multivariate_normal([xt; yt], competitor.mu(:, i), competitor.sigma(:, :, i)) / sum_gmm;
    
    % update values of [t-1] for M-step
    lambda = 1 - (1 - forget_a) / (forget_a * competitor.n_i + forget_b);
    competitor.W(i) = lambda ^ wti * competitor.W(i) + (1 - lambda ^ wti) / (1 - lambda) * 1;
    competitor.X(:,i) = lambda ^ wti * competitor.X(:,i) + (1 - lambda ^ wti) / (1 - lambda) * [xt; yt];
    competitor.XX(:, :, i) = lambda ^ wti * competitor.XX(:, :, i) + (1 - lambda ^ wti) / (1 - lambda) * [xt; yt] * [xt; yt]';
end

%%  M-step

sum_Wtj = sum(competitor.W);
for i = 1:num_g
    competitor.alpha(i) = competitor.W(i) / sum_Wtj;
    competitor.mu(:, i) = competitor.X(:, i) / competitor.W(i);
    competitor.sigma(:, :, i) = competitor.XX(:, :, i) / competitor.W(i) - (competitor.mu(:, i) * competitor.mu(:, i)');
    
    if ~isreal(competitor.sigma(:, :, i))
        disp('help')
    end
    if (sum(isnan(competitor.sigma(:, :, i))) ~= 0)
        disp('help')
    end
    
    %ensure that sigma  is PSD
    veig=eig(competitor.sigma(:, :, i));
    while min(veig)<0.000001
        reg_coef = 2;
        nn=0; 
        variance = trace(competitor.sigma(:, :, i))/(nn+1);
        variance = max(variance,0.01);
        competitor.sigma(:, :, i) = competitor.sigma(:, :, i) + reg_coef * variance^2 * eye(size(competitor.sigma(:, :, i)));
        veig = eig(competitor.sigma(:, :, i));
    end
end

competitor.n_i = competitor.n_i +1;
end