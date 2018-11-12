function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
    %LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
    %regression with multiple variables
    %   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
    %   cost of using theta as the parameter for linear regression to fit the 
    %   data points in X and y. Returns the cost in J and the gradient in grad
    %
    % Credit: Keivan Hassani Monfared, k1monfared@gmail.com and Andrew Ng, Coursera Team

    % Initialize some useful values
    m = length(y); % number of training examples

    %   Compute the cost and gradient of regularized linear 
    %   regression for a particular choice of theta.
    % 
    %   Set J to the cost and grad to the gradient.
    h = X * theta;
    J = (h-y)' * (h-y) / (2*m) + lambda/(2*m) * theta(2:end)' * theta(2:end);

    grad = (h-y)' * X / m;
    grad(2:end) = grad(2:end) + lambda/m * theta(2:end)';
    grad = grad(:);
end
