function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
%fprintf(' %f \n', size(X))
%fprintf(' %f \n', y)
%fprintf(' %f \n', theta)
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    distances = (X * theta) - y;
    theta_diff = alpha * (distances' * X) / m;
    theta = theta - theta_diff';

    % ============================================================

    %fprintf(' %f \n', X)
    %fprintf(' %f \n', y)
    %fprintf(' %f \n', theta)
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
