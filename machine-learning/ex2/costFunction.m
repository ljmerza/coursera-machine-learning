function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

sigmoid_logistic = sigmoid(X*theta);
ones_logistic = -y'*log(sigmoid_logistic);
zeros_logistic = (1-y)'*log(1 - sigmoid_logistic);

J = (1/m) * sum(ones_logistic - zeros_logistic);

% calc cost function derivative and resize matrix with zeros for X multiplication
copied_mat = repmat((sigmoid(X*theta) - y), 1, size(X,2));

% calc new gradient
grad = (1/m) * sum(X .* copied_mat);


% =============================================================

end
