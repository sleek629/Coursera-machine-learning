function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
  update = zeros(length(theta),1);
  for j = 1:length(theta)
    % for i = 1:m
      % update(j) += (theta' * X(i,:)' - y(i))*X(i,j);
    update(j) += ((theta' * X')' - y)'*X(:,j);
      % theta(j) -= alpha/m * (theta' * X(i,:)' - y(i))*X(i,j);
    % end
    % theta(j) -= alpha/m * update(j);
  end
  theta -= alpha/m * update;
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %











    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
