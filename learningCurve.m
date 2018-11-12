function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
    %LEARNINGCURVE Generates the train and cross validation set errors needed 
    %to plot a learning curve
    %   [error_train, error_val] = ...
    %       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
    %       cross validation set errors for a learning curve. In particular, 
    %       it returns two vectors of the same length - error_train and 
    %       error_val. Then, error_train(i) contains the training error for
    %       i examples (and similarly for error_val(i)).
    %
    %   Compute the train and test errors for
    %   dataset sizes from 1 up to m. In practice, when working with larger
    %   datasets, you might want to do this in larger intervals.
    %
    % Credit: Keivan Hassani Monfared, k1monfared@gmail.com and Andrew Ng, Coursera Team

    % Number of training examples
    m = size(X, 1);

    % preallocate
    error_train = zeros(m, 1);
    error_val   = zeros(m, 1);

    %   Return training errors in 
    %   error_train and the cross validation errors in error_val. 
    %   i.e., error_train(i) and 
    %   error_val(i) should give you the errors
    %   obtained after training on i examples.
    %
    % Note: Evaluating the training error on the first i training
    %       examples (i.e., X(1:i, :) and y(1:i)).
    %
    %       For the cross-validation error, instead evaluate on
    %       the _entire_ cross validation set (Xval and yval).
    %
    % Note: calling the function with the lambda argument set to 0. 
    
    for i = 1:m
        %   Compute train/cross validation errors using training examples 
        %   X(1:i, :) and y(1:i), storing the result in 
        %   error_train(i) and error_val(i)

        theta = trainLinearReg(X(1:i, :), y(1:i), lambda);
        error_train(i) =  linearRegCostFunction(X(1:i, :), y(1:i), theta, 0);
        error_val(i) =  linearRegCostFunction(Xval, yval, theta, 0);
    end
end
