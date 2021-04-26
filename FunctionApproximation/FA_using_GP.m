classdef FA_using_GP
    % Function Approximation using Gaussian Processes
    properties
        X_train
        X_test
        y_train
        signal_variance
        length_scale
        noise
        num_samples
    end
        
    methods
        
        function approx_func(obj)
            
            [f_mean, f_cov, stdv] = predictive_distribution(obj);
            
            posterior_samples = generate_samples(obj, f_mean, f_cov);
            
            plot_func(obj, posterior_samples, f_mean, stdv);
            
        end
            
        
        function covariance_matrix= get_covariance_matrix(obj, X_p, X_q)
            %
            % Built covariance matrix
            
            covariance_matrix = zeros(size(X_p, 2), size(X_q, 2));
            
            for i = 1:size(X_p, 2)
                for j = 1: size(X_q, 2)
                    covariance_matrix(i,j) = radial_basis_function(obj, X_p(i), X_q(j));
                end
            end
            
        end
        
        function samples = generate_samples(obj, f_mean, f_cov)
            samples = mvnrnd(f_mean,f_cov,obj.num_samples);
        end
        
        function [f_mean, f_cov, stdv] = predictive_distribution(obj)
            
            dim_X_train = size(obj.X_train,2);
            X_train_int = obj.X_train;
            X_test_int = obj.X_test;
            y_train_int = obj.y_train;
            
            K_traintrain = get_covariance_matrix(obj,X_train_int, X_train_int) + obj.noise^2 * eye(dim_X_train);
            
            K_traintest = get_covariance_matrix(obj,X_train_int, X_test_int);
            
            K_testtrain = get_covariance_matrix(obj,X_test_int, X_train_int);
            
            K_testtest = get_covariance_matrix(obj,X_test_int, X_test_int);
            
            f_mean = K_testtrain * inv(K_traintrain) * obj.y_train';
            
            f_cov = K_testtest - K_testtrain * inv(K_traintrain) * K_traintest;
            
            L_traintrain = chol(K_traintrain + obj.noise^2 * eye(dim_X_train));
            L_traintest = L_traintrain \ (L_traintrain' \ K_traintest);

            var = diag(K_testtest) - sum(L_traintest.^2, 1)';
            stdv = sqrt(var);
            
        end
        
        function covariance = radial_basis_function(obj,x_p, x_q)
            % Radial Basis Function calculating the covariance matrix
            
            squared_dist = x_p' * x_p - 2 * x_p' * x_q + x_q' * x_q;
            covariance = obj.signal_variance * exp( -1/ (2 * obj.length_scale^2) * squared_dist);
            
            
        end
        
        function plot_func(obj, posterior_samples, f_mean, stdv)
            y_test = sin(obj.X_test);
            
            y_upper = f_mean + 2*stdv;
            y_lower = f_mean - 2*stdv;
            
            figure
            title('Function Approximation')
            xlabel('x')
            ylabel('y')
            patch([obj.X_test fliplr(obj.X_test)], [y_upper' fliplr(y_lower')],'c', 'DisplayName', '2x Standard Deviation')
            hold on
            plot(obj.X_train, obj.y_train, 'x', 'DisplayName', 'Training Samples') 
            hold on
            plot(obj.X_test,y_test, 'LineWidth',1.5, 'Color','r', 'DisplayName', 'y = sin(x)')
            hold on
            plot(obj.X_test,posterior_samples, 'DisplayName', 'Samples from Predictive Distribution')
            hold on
            plot(obj.X_test,f_mean,'--','LineWidth', 1.5, 'Color', 'b', 'DisplayName', 'Approximated Mean')
            hold on
            ylim([-2 2])
            legend;
            
        end
    end
            
end