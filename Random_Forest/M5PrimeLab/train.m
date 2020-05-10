%% Clear and Close Figures
clear; close all; clc
%% Load Data
fprintf('Loading data ...\n');
X = load('C:\Users\ilCONDOR\Dropbox\unibz\Semester2\Machine_Learning\Project\repo\Logistic_Regression\titanic_numerical_clean.csv');
% X = X(randperm(size(X, 1)), :);
n = length(X(1,:))-1;
Y = X(:,n+1);
X = X(:,1:n);
features = zeros(n,1);
%==========================================================
%% Build model tree
fprintf('\nBuilding model tree ...\n');
params = m5pparams2('modelTree', true);
model = m5pbuild(X, Y, params, features);
m5pprint(model);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
%% Plot model tree
fprintf('\nPlotting model tree ...\n');
m5pplot(model, 'showNumCases', 'off', 'showSD', 0, 'precision', 1, 'dealWithNaN', false, 'layout', 'oblique', 'widthMult', 3, 'variableWidth', false, 'colorize', false, 'fontSize', 5); 
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
%% 10-fold Cross-Validation
fprintf('\n10-fold Cross-Validation on model tree...\n');
rng(1);
results = m5pcv(X, Y, params, features)
%==========================================================
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
%% Build regression tree
fprintf('\nBuilding regression tree ...\n');
params = m5pparams2('modelTree', false);
model = m5pbuild(X, Y, params, features);
m5pprint(model);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
%% Plot regression tree
fprintf('\nPlotting regression tree ...\n');
m5pplot(model,  'showNumCases', 'off', 'showSD', 0, 'precision', 1, 'dealWithNaN', false, 'layout', 'right', 'widthMult', 3, 'variableWidth', false, 'colorize', false, 'fontSize', 5); 
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
%% 10-fold Cross-Validation
fprintf('\n10-fold Cross-Validation on regression tree...\n');
rng(1);
results = m5pcv(X, Y, params, features)
%==========================================================
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
%% Predict
[Yq, contrib] = m5ppredict(model, X);
fprintf('\nPrediction: %f\n', Yq(1));
fprintf('Training set mean: %f\n', contrib(1,end));
fprintf('Input variable contributions:\n');
[~, idx] = sort(abs(contrib(1,1:end-1)), 'descend');
for i = idx
fprintf('x%d: %f\n', i, contrib(1,i));
end
%==========================================================
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
%% Extract decisions rules
fprintf('\nExtracting decisions rules: %f\n', Yq(1));
params = m5pparams2('modelTree', true, 'extractRules', 2);
model = m5pbuild(X, Y, params, features);
m5pprint(model);

%% 10-fold Cross-Validation on rules configuration
fprintf('\n10-fold Cross-Validation on rules configuration...\n');
rng(1);
results = m5pcv(X, Y, params, features)
%==========================================================
fprintf("=======END=========\n");