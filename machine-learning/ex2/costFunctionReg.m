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


sigmoid_logistic = sigmoid(X*theta);
ones_logistic = -y'*log(sigmoid_logistic);
zeros_logistic = (1-y)'*log(1 - sigmoid_logistic);

% skip the first theta - we don't regularize it
theta_to_regularize = theta(2:length(theta));

regularized = (lambda/(2*m)) * sum(theta_to_regularize .* theta_to_regularize);
J = ((1 / m) * sum(ones_logistic - zeros_logistic)) + regularized;


% calc cost function derivative and repeat matrix to resize for X multiplication
copied_mat = repmat((sigmoid(X*theta) - y), 1, size(X,2));
grad = (1/m) * sum(X .* copied_mat);

% only replace values after the first theta
grad(:,2:length(grad)) = grad(:,2:length(grad)) + (lambda/m)*theta_to_regularize';

% =============================================================

end
