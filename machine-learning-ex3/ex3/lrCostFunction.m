function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
theta_reg = theta;
theta_reg(1) = 0;

hypotheses = sigmoid(X*theta);
errorVector = hypotheses - y;
J = (1/m)*sum(-y .* log(hypotheses) - (1-y).*log(1-hypotheses)) + (lambda/(2*m))*theta_reg'*theta_reg;

grad=(1/m)*(X'*errorVector) + (lambda/m)*theta_reg;
% =============================================================

grad = grad(:);

end
