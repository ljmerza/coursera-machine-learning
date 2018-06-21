function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

    m = length(y); % number of training examples
    J_history = zeros(num_iters, 1); % where we store our gradient answers

    for iter = 1:num_iters
      
        hypothesis = X * theta;
        error = hypothesis - y;
        % error = ((X * theta) - y);
        
        gradient = (1/m) .* X' * error;
        % gradient = (1/m).* X' * ((X * theta) - y);
        
        theta = theta - alpha .* gradient;

        % Save the cost J in every iteration    
        J_history(iter) = computeCost(X, y, theta);

    end
end
