approximator = FA_using_GP;
approximator.X_train = [-5, -3, -1, 1, 3, 5];
approximator.X_test = linspace(-5,5,100);
approximator.y_train = sin(approximator.X_train);
approximator.signal_variance = 1;
approximator.length_scale = 1;
approximator.noise = 0;
approximator.num_samples = 2;

approximator.approx_func