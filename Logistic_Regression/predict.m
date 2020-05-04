function p = predict(theta, X)
% ====================== YOUR CODE HERE ======================

z = X*theta;
h = sigmoid(z);
m = length(X(:,1));
p= zeros(m,1);
p = h>=0.5;

% =========================================================================
end
