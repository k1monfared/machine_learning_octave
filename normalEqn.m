function [theta] = normalEqn(X, y)
    %NORMALEQN Computes the closed-form solution to linear regression 
    %   NORMALEQN(X,y) computes the closed-form solution to linear 
    %   regression using the normal equations.
    %
    % Credit: Keivan Hassani Monfared, k1monfared@gmail.com and Andrew Ng, Coursera Team

    % Compute the closed form solution to linear regression and put the result in theta.
    theta = pinv(X'*X) * X' * y;
end
