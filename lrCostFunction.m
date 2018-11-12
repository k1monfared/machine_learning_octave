function [J, grad] = lrCostFunction(theta, X, y, lambda)
    %LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
    %regularization
    %   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
    %   theta as the parameter for regularized logistic regression and the
    %   gradient of the cost w.r.t. to the parameters. 
    %
    % Credit: Keivan Hassani Monfared, k1monfared@gmail.com and Andrew Ng, Coursera Team

    % Initialize
    m = length(y); % number of training examples

    %   Compute the cost of a particular choice of theta.
    %   You should set J to the cost.
    %   Compute the partial derivatives and set grad to the partial
    %   derivatives of the cost w.r.t. each parameter in theta
    %
    %       Each row of the resulting matrix will contain the value of the
    %       prediction for that example.

    h = sigmoid(X * theta);
        J = - (y' * log(h) + (1-y)' * log(1-h)) / m + lambda/(2*m) * theta(2:end)' * theta(2:end);

    grad = (h-y)' * X / m;
    grad(2:end) = grad(2:end) + lambda/m * theta(2:end)';
    grad = grad(:);
end
