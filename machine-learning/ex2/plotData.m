function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% getting all the indexes of 1/0 from the y data set
ones_indexes = find(y==1); 
zeros_indexes = find(y == 0);

% plot ones - first get all indexes from X that have y=1 
% (first and second columns for x/y coordinates) -> X(ones_indexes, 1), X(ones_indexes, 2)
ones_x = X(ones_indexes, 1)
ones_y = X(ones_indexes, 2)
plot(ones_x, ones_y, 'k+','LineWidth', 2, 'MarkerSize', 10);

% plot zeros
plot(X(zeros_indexes, 1), X(zeros_indexes, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 10);




% =========================================================================



hold off;

end
