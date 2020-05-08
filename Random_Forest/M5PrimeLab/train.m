%% Clear and Close Figures
clear; close all; clc

%% Load Data
fprintf('Loading data ...\n');
X = load('C:\Users\ilCONDOR\Dropbox\unibz\Semester2\Machine_Learning\Project\repo\Logistic_Regression\titanic_numerical_clean.csv');
n = length(X(1,:));
Y = X(:,n);
features = zeros(n,1);
%==========================================================

%% Build and print model tree
fprintf('\nBuilding and printing model tree ...\n');
params = m5pparams2('modelTree', true);
model = m5pbuild(X, Y, params, features);
m5pprint(model);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Plot model tree
fprintf('\nPlotting model tree ...\n');
m5pplot(model); 
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% 10-fold Cross-Validation
fprintf('\n10-fold Cross-Validation ...\n');
rng(1);
results = m5pcv(X, Y, params, features)
%==========================================================
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Build and print regression tree
fprintf('\nBuilding and printing regression tree ...\n');
params = m5pparams2('modelTree', false);
model = m5pbuild(X, Y, params, features);
m5pprint(model);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% 10-fold Cross-Validation
fprintf('\n10-fold Cross-Validation ...\n');
rng(1);
results = m5pcv(X, Y, params, features)
%==========================================================
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Print precision tree
fprintf('\nPrinting precision tree ...\n');
m5pplot(model, 'precision', 3, 'layout', 'right');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Predict
P = ones(1,n);
[Yq, contrib] = m5ppredict(model, P);
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
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% 10-fold Cross-Validation
fprintf('\n10-fold Cross-Validation ...\n');
rng(1);
results = m5pcv(X, Y, params, features)
%==========================================================

fprintf("=======END=========\n");