function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
    %GRADIENTDESCENT Performs gradient descent to learn theta
    %   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
    %   taking num_iters gradient steps with learning rate alpha
    %
    % Credit: Keivan Hassani Monfared, k1monfared@gmail.com and Andrew Ng, Coursera Team

    % Initialize
    m = length(y); % number of training examples
    J_history = zeros(num_iters, 1);

    for iter = 1:num_iters
        % Perform a single gradient step on the parameter vector theta. 
        h = X * theta;
        delta = X' * (h-y)/m;
        theta = theta - alpha * delta;

        % Save the cost J in every iteration    
        J_history(iter) = computeCost(X, y, theta);
    end
end
