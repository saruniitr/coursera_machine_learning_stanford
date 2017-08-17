function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

opt_C = 0;
opt_sigma = 0;
error = 1000000;    % some big number which will be replaced with min value in
					% each iteration

% loop over all combinations of C and sigma
for i = 1:length(C_values)
    for j = 1:length(sigma_values)
	    current_C = C_values(i);
	    current_sigma = sigma_values(j);

		model = svmTrain(X, y, current_C,
						 @(x1, x2) gaussianKernel(x1, x2, current_sigma));
		predictions = svmPredict(model, Xval);

		new_error = mean(double(predictions ~= yval));

		% if the current error is less than previous error then replace it and
		% update new values of C and sigma
		if new_error < error
		    error = new_error;
		    opt_C = current_C;
			opt_sigma = current_sigma;
		end
	end
end

% return the optimal values of C and sigma
C = opt_C;
sigma = opt_sigma;


% =========================================================================

end
