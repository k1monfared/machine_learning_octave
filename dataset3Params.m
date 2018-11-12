function [C, sigma] = dataset3Params(X, y, Xval, yval)
    %DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
    %where you select the optimal (C, sigma) learning parameters to use for SVM
    %with RBF kernel
    %   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
    %   sigma. You should complete this function to return the optimal C and 
    %   sigma based on a cross-validation set.
    %
    % Credit: Keivan Hassani Monfared, k1monfared@gmail.com and Andrew Ng, Coursera Team

    %   Fill in this function to return the optimal C and sigma
    %   learning parameters found using the cross validation set.
    %   You can use svmPredict to predict the labels on the cross
    %   validation set. For example, 
    %       predictions = svmPredict(model, Xval);
    %   will return the predictions on the cross validation set.

    error = inf;

    for i = -1:1
        temp_C = 10^i;
        for j = -2:0
            temp_sigma = 10^j;
            model = svmTrain(X, y, temp_C, @(x1, x2) gaussianKernel(x1, x2, temp_sigma)); 
            predictions = svmPredict(model, Xval);
            temp_error = mean(double(predictions ~= yval));
            if temp_error < error
                C = temp_C;
                sigma = temp_sigma;
                error = temp_error;
            end
        end
    end
end
