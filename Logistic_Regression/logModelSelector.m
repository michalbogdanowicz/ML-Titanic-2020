%% Clear and Close Figures
clear; close all; clc

fprintf('Loading data ...\n');
%% Load Data
data = load('titanic_numerical_clean.csv');

%==========================================================
num_columns = columns(data)
y = data(:, num_columns);
k = 10; # <---- K VALUE

c = cvpartition(y, 'KFold', k)
mean_accuracy = 0;

# need to do in total 5 times for power that comes from x^1 to x^10.
for power_iteration = 1 : 5
      fprintf('\n MAX POWER == %f\n', power_iteration)

  for iteration = 1:k
  
    train_l_vector = training (c, iteration); % test training vector
    test_l_data = test (c, iteration); % test logic vector 

    train_data = data(train_l_vector, :);
    test_data = data(test_l_data, :);

    train_X = train_data(:, 1:num_columns-1);
    train_y = train_data(:, num_columns);
    
    m = length(train_y);

    X = generateModel(train_X, power_iteration);
    
    t_init = zeros(length(X(1,:)),1);

    [cost, grad] = costFunction(t_init, X,train_y);

    % set the options for the algorithm fminunc
    options = optimset( 'GradObj','on','MaxIter' , 400);
    % call fminunc
    [theta, cost] = fminunc(@(t)(costFunction(t, X, train_y)), t_init, options);

    test_y = test_data(:, num_columns);

    m = length(test_y);
    test_X = test_data(:, 1:num_columns-1);
    X = generateModel(test_X, power_iteration);
    
    h = predict(theta,X);
    acc = sum(h == test_y)/length(test_y);
    fprintf('Test Accuracy: %f\n', acc)
    mean_accuracy = mean_accuracy + acc;
  endfor
    fprintf('mean accuracy == %f\n', mean_accuracy / k)
    fflush(stdout)
    mean_accuracy = 0;
endfor

#plotBoundry(theta,X,y,0);

% fprintf('\nProgram paused. Press enter to continue.\n');
%%%%========================================================================

% XX = [X X(:,2).^2 X(:,3).^2 X(:,2).*X(:,3)];
% t_init = zeros(length(XX(1,:)),1);
% lambda = 1;
% [theta1, J] = fminunc(@(t)(costFunctionReg(t, XX, y, lambda)), t_init, options);
% hh =predict(theta1,XX);
% fprintf('Regularized Train Accuracy: %f\n', sum(hh == y)/length(y));
%%%========================================================================
% plotBoundry(theta1,X,y,1);
fprintf("=======END=========\n");