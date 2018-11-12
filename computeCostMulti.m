function J = computeCostMulti(X, y, theta)
    %COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
    %   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
    %   parameter for linear regression to fit the data points in X and y
    %
    % Credit: Keivan Hassani Monfared, k1monfared@gmail.com and Andrew Ng, Coursera Team

    % Initialize some useful values
    m = length(y); % number of training examples

    % Compute the cost of a particular choice of theta
    h = X * theta;
    J = (h-y)' * (h-y) / (2*m);
end
