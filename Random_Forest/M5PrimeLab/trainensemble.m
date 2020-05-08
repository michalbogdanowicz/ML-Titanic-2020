%% Clear and Close Figures
clear; close all; clc

%% Load Data
fprintf('Loading data ...\n');
X = load('C:\Users\ilCONDOR\Dropbox\unibz\Semester2\Machine_Learning\Project\repo\Logistic_Regression\titanic_numerical_clean.csv');
n = length(X(1,:));
Y = X(:,n);
features = zeros(n,1);
%==========================================================

%% Growing ensembles of trees
fprintf('\nGrowing ensembles of trees ...\n');
params = m5pparams(false, 1, 4, false, 0, 0.05);

nTrees = 10;
numVarsTry = [-1,-1];
paramsEnsemble = m5pparamsensemble(nTrees, numVarsTry, true, 1, false, true, 1, false, 0);

figure; hold on;
for i = 1:length(numVarsTry)
paramsEnsemble.numVarsTry = numVarsTry(i);
[~, ~, ensembleResults] = m5pbuild(X, Y, params, features, paramsEnsemble);
plot(ensembleResults.OOBError(:,1));
end
grid on;
xlabel('Number of trees');
ylabel('Out-of-bag MSE');
legend('alive', 'dead');

paramsEnsemble = m5pparamsensemble(nTrees);
[model, time, ensembleResults] = m5pbuild(X, Y, params, features, paramsEnsemble);
%==========================================================
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Plot graph
fprintf('\nPlotting ensembles MSE ...\n');
figure;
plot(ensembleResults.OOBError(:,1));
grid on;
xlabel('Number of trees');
ylabel('Out-of-bag MSE');

%% Plot bar
fprintf('\nPlotting variable importance ...\n');
figure;
bar(ensembleResults.varImportance(3,:) ./ ensembleResults.varImportance(4,:));
xlabel('Variable number');
ylabel('Variable importance');

%% Forest Floor main effect plots
%% figure;
%% contrib = ensembleResults.OOBContrib;
%% cminmax = [min(min(contrib(:,1:(end-1))))-0.5 max(max(contrib(:,1:(end-1))))+0.5];
%% for i = 1 : size(X,2)
%% subplot(n,5,i);
%% scatter(X(:,i), contrib(:,i), 50, '.');
%% ylim(cminmax); xlim([min(X(:,i)) max(X(:,i))]);
%% xlabel(['x_{' num2str(i) '}']); box on;
%% end
%==========================================================
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Predict
fprintf('\nPredicting ensembles of trees ...\n');
P = ones(1,n);
[Yq, contrib] = m5ppredict(model, P);
fprintf('Prediction: %f\n', Yq(1));
fprintf('In-bag mean: %f\n', contrib(1,end));
fprintf('Input variable contributions:\n');
[~, idx] = sort(abs(contrib(1,1:end-1)), 'descend');
for i = idx
fprintf('x%d: %f\n', i, contrib(1,i));
end
%==========================================================
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% 10-fold Cross-Validation
fprintf('\n10-fold Cross-Validation ...\n');
rng(1);
resultsCV = m5pcv(X, Y, params, features, 10, [], [], paramsEnsemble);
figure;
plot(ensembleResults.OOBError(:,1));
hold on;
plot(resultsCV.MSE);
grid on;
xlabel('Number of trees');
ylabel('MSE');
legend('alive', 'dead');
%==========================================================
 
fprintf("\n=======END=========\n");