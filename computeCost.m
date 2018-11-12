function J = computeCost(X, y, theta)
    %COMPUTECOST Compute cost for linear regression
    %   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    %   parameter for linear regression to fit the data points in X and y
    %
    % Credit: Keivan Hassani Monfared, k1monfared@gmail.com and Andrew Ng, Coursera Team

    % Initialize some useful values
    m = length(y); % number of training examples

    %   Compute the cost of a particular choice of theta
    %   You should set J to the cost.
    h = X * theta - y;
    J = h' * h / (2*m);

end
