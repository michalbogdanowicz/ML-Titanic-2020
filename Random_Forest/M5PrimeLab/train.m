%% Clear and Close Figures
clear; close all; clc

%% Load Data
fprintf('Loading data ...\n');
X = load('C:\Users\ilCONDOR\Dropbox\unibz\Semester2\Machine_Learning\Project\repo\Logistic_Regression\titanic_numerical_clean.csv');
Y = X(:,11);
isBinCat = [false false false false false false false false false false false];
%==========================================================

%% Build and print model tree
fprintf('Building and printing model tree ...\n');
params = m5pparams2('modelTree', true);
model = m5pbuild(X, Y, params, isBinCat);
m5pprint(model);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Plot model tree
fprintf('Plotting model tree ...\n');
m5pplot(model); 
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% 10-fold Cross-Validation
fprintf('10-fold Cross-Validation ...\n');
rng(1);
results = m5pcv(X, Y, params, isBinCat)
%==========================================================
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Build and print regression tree
fprintf('Building and printing regression tree ...\n');
params = m5pparams2('modelTree', false);
model = m5pbuild(X, Y, params, isBinCat);
m5pprint(model);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% 10-fold Cross-Validation
fprintf('10-fold Cross-Validation ...\n');
rng(1);
results = m5pcv(X, Y, params, isBinCat)
%==========================================================
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Print precision tree
fprintf('Printing precision tree ...\n');
m5pplot(model, 'precision', 3, 'layout', 'right');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Predict
[Yq, contrib] = m5ppredict(model, [0.5 0 2 0 0 0 0 0 0 0 0]);
fprintf('Prediction: %f\n', Yq(1));
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
fprintf('Extracting decisions rules: %f\n', Yq(1));
params = m5pparams2('modelTree', true, 'extractRules', 2);
model = m5pbuild(X, Y, params, isBinCat);
m5pprint(model);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% 10-fold Cross-Validation
fprintf('10-fold Cross-Validation ...\n');
rng(1);
results = m5pcv(X, Y, params, isBinCat)
%==========================================================
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Growing ensembles of trees
fprintf('1Growing ensembles of trees ...\n');
params = m5pparams(false, 1, 5, false, 0, 1E-6);

paramsEnsemble = m5pparamsensemble(50, [], [], [], [], true, 0, false);
numVarsTry = [1,1];
figure; hold on;
for i = 1:2
paramsEnsemble.numVarsTry = numVarsTry(i);
[~, ~, ensembleResults] = m5pbuild(X, Y, params, isBinCat, paramsEnsemble);
plot(ensembleResults.OOBError(:,1));
end
grid on;
xlabel('Number of trees');
ylabel('Out-of-bag MSE');
legend('alive', 'dead');

paramsEnsemble = m5pparamsensemble(50);
[model, time, ensembleResults] = m5pbuild(X, Y, params, isBinCat, paramsEnsemble);

%% Plot graph
fprintf('Plotting ensembles of trees ...\n');
figure;
plot(ensembleResults.OOBError(:,1));
grid on;
xlabel('Number of trees');
ylabel('Out-of-bag MSE');

%% Plot bar
fprintf('Plotting ensembles of trees (bar) ...\n');
figure;
bar(ensembleResults.varImportance(3,:) ./ ensembleResults.varImportance(4,:));
xlabel('Variable number');
ylabel('Variable importance');
figure;
contrib = ensembleResults.OOBContrib;
cminmax = [min(min(contrib(:,1:(end-1))))-0.5 max(max(contrib(:,1:(end-1))))+0.5];
for i = 1 : size(X,2)
subplot(3,5,i);
scatter(X(:,i), contrib(:,i), 50, '.');
ylim(cminmax); xlim([min(X(:,i)) max(X(:,i))]);
xlabel(['x_{' num2str(i) '}']); box on;
end

%% Predict
fprintf('Predicting ensembles of trees ...\n');
[Yq, contrib] = m5ppredict(model, [0 0 0 0 0 0 0 0 0 0 0]);
fprintf('Prediction: %f\n', Yq(1));
fprintf('In-bag mean: %f\n', contrib(1,end));
fprintf('Input variable contributions:\n');
[~, idx] = sort(abs(contrib(1,1:end-1)), 'descend');
for i = idx
fprintf('x%d: %f\n', i, contrib(1,i));
end

%% 10-fold Cross-Validation
fprintf('10-fold Cross-Validation ...\n');
rng(1);
resultsCV = m5pcv(X, Y, params, isBinCat, 10, [], [], paramsEnsemble);
figure;
plot(ensembleResults.OOBError(:,1));
hold on;
plot(resultsCV.MSE);
grid on;
xlabel('Number of trees');
ylabel('MSE');
legend({'Out-of-bag' 'Cross-Validation'}, 'Location', 'NorthEast');
 
fprintf("=======END=========\n");