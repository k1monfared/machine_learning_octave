%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

X = mapFeature(X(:,1), X(:,2));

% Set regularization parameter lambda to 1
n = 17;
for t = 1:n
    % Initialize fitting parameters
    initial_theta = zeros(size(X, 2), 1);

    % Set regularization parameter lambda to 1 (you should vary this)
    lambda = 10^(-(t-(n-3)/2)/2);

    % Set Options
    options = optimset('GradObj', 'on', 'MaxIter', 400);

    % Optimize
    [theta, J, exit_flag] = ...
        fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
    % Compute accuracy on our training set
    p = predict(theta, X);
    
    % Plot Boundary
    plotDecisionBoundary(theta, X, y);
    hold on;
    title([sprintf('lambda = %0.5f Train Accuracy: %0.0f', lambda, mean(double(p == y)) * 100)]);

    % Labels and Legend
    xlabel('Microchip Test 1')
    ylabel('Microchip Test 2')

    legend('y = 1', 'y = 0', 'Decision boundary')
    hold off;
    
    
    %print animation.pdf -append
end 
im = imread ("animation.pdf", "Index", "all");
imwrite (im, "animation.gif", "DelayTime", .5)
