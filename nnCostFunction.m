function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
    %NNCOSTFUNCTION Implements the neural network cost function for a two layer
    %neural network which performs classification
    %   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    %   X, y, lambda) computes the cost and gradient of the neural network. The
    %   parameters for the neural network are "unrolled" into the vector
    %   nn_params and need to be converted back into the weight matrices. 
    % 
    %   The returned parameter grad will be a "unrolled" vector of the
    %   partial derivatives of the neural network.
    %
    % Credit: Keivan Hassani Monfared, k1monfared@gmail.com and Andrew Ng, Coursera Team

    % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    % for our 2 layer neural network
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));

    % Setup some useful variables
    m = size(X, 1);
    X = [ones(m,1), X];

    % Feedforward the neural network and return the cost in the
    % variable J.
    yy = zeros(size(Theta2,1),m);
    for i = 1:size(y,1)
        yy(y(i),i) = 1;
    end

    Z2 = X * Theta1';
    a2 = sigmoid(Z2);
    a2 = [ones(m,1), a2];
    Z3 = a2 * Theta2';
    a3 = sigmoid(Z3);
    J = - sum(sum((yy' .* log(a3) + (1-yy)' .* log(1-a3)))) / m;

    % Backpropagation algorithm to compute the gradients
    % Theta1_grad and Theta2_grad. Return the partial derivatives of
    % the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    % Theta2_grad, respectively.
    % 
    % Note: The vector y passed into the function is a vector of labels
    %       containing values from 1..K. Map this vector into a 
    %       binary vector of 1's and 0's to be used with the neural network
    %       cost function.

    delta3 = a3 - yy';
    delta2 = delta3 * Theta2(:,2:end) .* sigmoidGradient(Z2);

    Delta2 = delta3' * a2;
    Delta1 = delta2' * X;

    reg_theta1 = [zeros(size(Theta1,1),1), Theta1(:,2:end)];
    reg_theta2 = [zeros(size(Theta2,1),1), Theta2(:,2:end)];

    Theta1_grad = Delta1/m + lambda/m * reg_theta1;
    Theta2_grad = Delta2/m + lambda/m * reg_theta2;

    % Regularization with the cost function and gradients.
    % Compute the gradients for the regularization separately and then add them 
    % to Theta1_grad and Theta2_grad from above.
    Reg =  lambda/(2*m) * (sum(sum(Theta1(:,2:end) .^2)) + sum(sum(Theta2(:,2:end) .^2)));
    J = J + Reg;

    % Unroll gradients
    grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
