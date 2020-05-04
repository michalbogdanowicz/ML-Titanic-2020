function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

predictions = sigmoid(X*theta);
reg_term = (lambda/(2*m)) * sum(theta(2:end).^2);
calcErrors = -y.*log(predictions) - (1 -y).*log(1-predictions);
J = (1/m)*sum(calcErrors)+reg_term;

% prepend a 0 column to our reg_term matrix so we can use simple matrix addition
reg_term = [0 (lambda*theta(2:end)/m)'];
grad = sum(X.*(predictions - y)) / m + reg_term;
% =============================================================

end