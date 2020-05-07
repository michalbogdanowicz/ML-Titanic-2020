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
for iteration = 1:k
  
  train_l_vector = training (c, iteration); % test training vector
  test_l_data = test (c, iteration); % test logic vector 

  train_data = data(train_l_vector, :);
  test_data = data(test_l_data, :);

  train_X = train_data(:, 1:num_columns-1);
  train_y = train_data(:, num_columns);
  
  m = length(train_y);
  t_init=zeros(num_columns,1);

  train_X = [ones(m,1) train_X]; %this should be the bias.
  [cost, grad] = costFunction(t_init, train_X,train_y);

  % fprintf('Cost at initial theta (zeros): %f\n', cost);
  % fprintf('Gradient at initial theta (zeros): \n');
  % fprintf(' %f \n', grad);

  % set the options for the algorithm fminunc
  options = optimset( 'GradObj','on','MaxIter' , 400);
  % call fminunc
  [theta, cost] = fminunc(@(t)(costFunction(t, train_X, train_y)), t_init, options);

  % fprintf('Cost after fminunc at initial theta (zeros): %f\n', cost);
  % fprintf('Gradient after fminunc at initial theta (zeros): \n');
  % fprintf(' %f \n', grad);
  
  %%gradient
  
  % need to predict on test now with a bias.
  test_y = test_data(:, num_columns);

  m = length(test_y);
  test_X = test_data(:, 1:num_columns-1);
  test_X = [ones(m,1) test_X]; %this should be the bias.

  
  h = predict(theta,test_X);
  acc = sum(h == test_y)/length(test_y);
  fprintf('Train Accuracy: %f\n', acc)
  mean_accuracy = mean_accuracy + acc;
endfor
  fprintf('\n mean accuracy ==: %f\n', mean_accuracy / k)

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