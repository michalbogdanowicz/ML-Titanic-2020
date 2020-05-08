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
mean_precision = 0;
mean_f1 = 0;
mean_recall = 0;
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
    % make a matrix full of ones of the same size of the h or test_y
    % get the positives 
    true_positive = sum( test_y(test_y == 1) == h(test_y == 1));
    false_positive = sum( test_y(h == 1) != h(h == 1));
    false_negative = sum( test_y(h == 0) != h(h == 0));
    % precision. 
    % True positive / (True Positive + False Positive)
    measure_precision = true_positive/( true_positive + false_positive);
    % Recall
    % True Positive / (True positive + False Negative)
    recall = true_positive / (true_positive + false_negative);
    %F1-measure
    % 2 (Recall * Precision) / (recall + precision)
    f1 =  (2 * recall * measure_precision) / (recall + measure_precision);

    fprintf('Test %d Accuracy: %f\n', iteration ,acc)
    mean_accuracy = mean_accuracy + acc;
    mean_precision = mean_precision + measure_precision;
    mean_f1 = mean_f1 + f1;
    mean_recall = mean_recall + recall;
  endfor
    fprintf('mean accuracy == %f\n', mean_accuracy / k)
    fprintf('mean precision == %f\n', mean_precision / k)
    fprintf('mean recall == %f\n', mean_recall / k)
    fprintf('mean f1 == %f\n', mean_f1 / k)
    fflush(stdout)
    mean_accuracy = 0;
    mean_precision = 0;
    mean_f1 = 0;
    mean_recall = 0;
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