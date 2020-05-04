%% Machine Learning Assignment 1  Logistic Regression
%
%% ============ Code HERE ============

%You code should Load the data
%Initialize the parameters (0 for thetas, lamda=1, etc)
% test each of the developed functions, outputting some examples 
% you can refer to the main function of the linear regression assignment lreg.m

%% Clear and Close Figures
clear; close all; clc

fprintf('Loading data ...\n');
%% Load Data
data = load('students.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
t_init=zeros(3,1);
%==========================================================
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

X = [ones(m,1) X];
[cost, grad] = costFunction(t_init, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

% set the options for the algorithm fminunc
options = optimset( 'GradObj','on','MaxIter' , 400);
% call fminunc
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), t_init, options);


fprintf('Cost after fminunc at initial theta (zeros): %f\n', cost);
fprintf('Gradient after fminunc at initial theta (zeros): \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
h = predict(theta,X);
fprintf('Train Accuracy: %f\n', sum(h == y)/length(y));
plotBoundry(theta,X,y,0);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
%%%%========================================================================

XX = [X X(:,2).^2 X(:,3).^2 X(:,2).*X(:,3)];
t_init = zeros(length(XX(1,:)),1);
lambda = 1;
[theta1, J] = fminunc(@(t)(costFunctionReg(t, XX, y, lambda)), t_init, options);
hh =predict(theta1,XX);
fprintf('Regularized Train Accuracy: %f\n', sum(hh == y)/length(y));
%%%========================================================================
plotBoundry(theta1,X,y,1);
fprintf("=======END=========\n");