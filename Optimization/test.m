clear all
clc

findExtrema = FindExtrema;
findExtrema.max_iteration = 20;
findExtrema.decay = true;
findExtrema.min_learning_rate=0.001;


%% Gradient Descent starting at x_init = 0
findExtrema.x_init = 0;
findExtrema.learning_rate = 1.2;
findExtrema.decay_rate = 0.3;
[x_min, objFunc_min, iteration] = findExtrema.gradientDescent;
x_min
objFunc_min
iteration

%% Gradient Descent starting at x_init = 1
findExtrema.x_init = 1;
findExtrema.learning_rate = 1.4;
findExtrema.decay_rate = 0.0;
[x_min, objFunc_min, iteration] = findExtrema.gradientDescent;
x_min
objFunc_min
iteration

%% Gradient Descent starting at x_init = 2
findExtrema.x_init = 2;
findExtrema.learning_rate = 0.03;
findExtrema.decay_rate = 0.001;
[x_min, objFunc_min, iteration] = findExtrema.gradientDescent;
x_min
objFunc_min
iteration

%% Gradient Ascent starting at x_init = 0
findExtrema.x_init = 0;
findExtrema.learning_rate = 0.4;
findExtrema.decay_rate = 0.1;
[x_max, objFunc_max, iteration] = findExtrema.gradientAscent;
x_max
objFunc_max
iteration

%% Gradient Ascent starting at x_init = 1
findExtrema.x_init = 1;
findExtrema.learning_rate = 1.4;
findExtrema.decay_rate = 0.11;
[x_max, objFunc_max, iteration] = findExtrema.gradientAscent;
x_max
objFunc_max
iteration

%% Gradient Ascent starting at x_init = 2
findExtrema.x_init = 2;
findExtrema.learning_rate = 1.2;
findExtrema.decay_rate = 0.1;
[x_max, objFunc_max, iteration] = findExtrema.gradientAscent;
x_max
objFunc_max
iteration