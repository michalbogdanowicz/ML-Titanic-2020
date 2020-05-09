%% Clear and Close Figures
clear; close all; clc
%% declaration of variables
mean_accuracy = 0;
mean_precision = 0;
mean_f1 = 0;
mean_recall = 0;
mean_mae = 0;
mean_mse = 0;
mean_rae = 0;
mean_rse= 0;
mean_mae_training = 0;

graph_data_CV_training_ERROR = zeros(5,2);

fprintf('Loading data ...\n');
%% Load Data
data = load('titanic_numerical_clean.csv');
% Randomize the rows. The data is ordered by something. And this created "unfair" folds.
data = data(randperm(size(data, 1)), :);
%==========================================================
num_columns = columns(data);
y = data(:, num_columns);
k = 10; # <---- K VALUE

c = cvpartition(y, 'KFold', k)
training_examples_vector = [5,10,25,50,75,100,125,135,150,165,172,190]
DATA_FOR_LEARNING_CURVE = zeros(3,columns(training_examples_vector));
plottable_error_on_pol_degree = zeros(2,5);
index = 0;
fprintf('\n MAX POWER == %f\n', 2)

# need to do in total 5 times for power that comes from x^1 to x^10.
for training_examples = training_examples_vector
index = index + 1;
DATA_FOR_LEARNING_CURVE(1,index) = training_examples_vector(index);

  for iteration = 1:k
  
    train_l_vector = training (c, iteration); % test training vector
    test_l_data = test (c, iteration); % test logic vector 

    train_data = data(train_l_vector, :);
    test_data = data(test_l_data, :);

    train_X = train_data(:, 1:num_columns-1);
    train_y = train_data(:, num_columns);
    
    m = length(train_y);

    X = generateModel(train_X, 2);
    
    t_init = zeros(length(X(1,:)),1);

    [cost, grad] = costFunction(t_init, X,train_y);

    % set the options for the algorithm fminunc
    options = optimset( 'GradObj','on','MaxIter',  training_examples, 'TolX', 0);
    % call fminunc
    [theta, cost, info] = fminunc(@(t)(costFunction(t, X, train_y)), t_init, options);
    fprintf('the info is  %f', info )
    hypothesis_training = predict(theta,X);
    mae_training = sum(abs(train_y - hypothesis_training))/length(train_y);
    
    test_y = test_data(:, num_columns);

    m = length(test_y);
    test_X = test_data(:, 1:num_columns-1);
    %% values in the test_X that have only 0 or 1 will not be elevated to a power.
    %% as it would just duplicate the same exact feature.
    
    test_model_X = generateModel(test_X, 2);
    
    h = predict(theta,test_model_X);
    
    
    %%%% MEASURES START %%%%
    acc = sum(h == test_y)/length(test_y);
    
    
    % Mean Absolute Error
    mae = sum(abs(test_y - h))/length(test_y);
    % Mean Squared Error
    mse = sum( (test_y - h).^2)/length(test_y);
    % Relative Absolute Error
    ground_truth_mean = sum(test_y)/length(test_y);
    rae = sum( abs(test_y - h))/sum(abs(test_y - ground_truth_mean));
    % Relative Squared Error
    rse = sum( (test_y - h).^2)/sum((test_y - ground_truth_mean).^2);

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
    
    %fprintf('Test %d Accuracy: %f\n', iteration ,acc)
    mean_accuracy = mean_accuracy + acc;
    mean_precision = mean_precision + measure_precision;
    mean_f1 = mean_f1 + f1;
    mean_recall = mean_recall + recall;
    mean_mae = mean_mae + mae;
    mean_mse = mean_mse + mse;
    mean_rae = mean_rae + rae;
    mean_rse = mean_rse + rse;
    mean_mae_training = mean_mae_training + mae_training;

    %%%% MEASURES END %%%%

endfor
    fprintf('mean values across folds for training_amount ->  %f\n', training_examples)
     printf('training mae == %.3f\n', mean_mae_training / k)
    fprintf('accuracy == %.3f\n', mean_accuracy / k)
    fprintf('precision == %.3f\n', mean_precision / k)
    fprintf('recall == %.3f\n', mean_recall / k)
    fprintf('f1 == %.3f\n', mean_f1 / k)
    fprintf('mae == %.3f\n', mean_mae / k)
    fprintf('mse == %.3f\n', mean_mse / k)
    fprintf('rae == %.3f\n', mean_rae / k)
    fprintf('rse == %.3f\n', mean_rse / k)
    DATA_FOR_LEARNING_CURVE(2,index) = mean_mae_training/ k;
    DATA_FOR_LEARNING_CURVE(3,index) =  mean_mae /k ;
    fflush(stdout);
    DATA_FOR_LEARNING_CURVE
    mean_accuracy = 0;
    mean_precision = 0;
    mean_f1 = 0;
    mean_recall = 0;
    mean_mae = 0;
    mean_mse = 0;
    mean_rae = 0;
    mean_rse= 0;
    mean_mae_training= 0;
endfor

fprintf('second row is the training mae')
fprintf('third row is the cv mae')

%plot([1,2,3,4,5],graph_data_CV_training_ERROR(1,:), [1,2,3,4,5],graph_data_CV_training_ERROR(2,:))

%plotBoundry(theta,X,y,0);

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