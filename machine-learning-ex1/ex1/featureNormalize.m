function [X_norm, mu, sigma] = featureNormalize(X)
  X_norm = X;
  mu = zeros(1, size(X, 2));
  sigma = zeros(1, size(X, 2));
  
  num_features = size(X,2);
  
  for x=1:num_features,
    mu(x) = mean(X(:,x));
    sigma(x) = std(X(:,x));
    
    X_norm(:,x) = (X_norm(:,x)-mu(x)) / sigma(x);
  end
  
end
