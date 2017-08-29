%%
% Machine Learning - Logistic Regression
%
% Credit Card Fraud Analysis This program processes Credit card transaction data
% and classifies whether a given transaction is fradulent or normal using
% Logistic Regression. The dataset is a standard dataset taken from internet. It
% is available on Kaggle and can also be downloaded from various Machine
% learning blogs etc.
%
% source: https://www.kaggle.com/dalpozz/creditcardfraud The datasets contains
% transactions made by credit cards in September 2013 by european
% cardholders. This dataset presents transactions that occurred in two days,
% where we have 492 frauds out of 284,807 transactions. The dataset is highly
% unbalanced, the positive class (frauds) account for 0.172% of all
% transactions.

% It contains only numerical input variables which are the result of a PCA
% transformation. Unfortunately, due to confidentiality issues, we cannot
% provide the original features and more background information about the
% data. Features V1, V2, ... V28 are the principal components obtained with
% PCA, the only features which have not been transformed with PCA are 'Time' and
% 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction
% and the first transaction in the dataset. The feature 'Amount' is the
% transaction Amount, this feature can be used for example-dependant cost-
% senstivelearning. Feature 'Class' is the response variable and it takes value
% 1 in case of fraud and 0 otherwise.
%
% Citation: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca
% Bontempi. Calibrating Probability with Undersampling for Unbalanced
% Classification. In Symposium on Computational Intelligence and Data Mining
% (CIDM), IEEE, 2015
%
% Note: This code contains some helper functions that are implemented as part of
% programming exercises in Machine Learning Specialization course from Coursera
% taught by Prof. Andrew Ng.
%
%%

clear all;

data = csvread('./data/creditcard.csv');

[dm, dn] = size(data);

% split data into training and validation set (70-30%)
split = 0.7;

% shuffle the data before splitting into training and validation set
data_shuffle = data(randperm(dm), :);

% separate out 'Class' column
ytrain = data_shuffle(1:round(dm * split), 31);
yval = data_shuffle((round(dm * split) + 1):end, 31);

% Extract all the features
Xshuffle = data_shuffle(:, 1:30);

# Apply Normalization to features and split
[Xnorm mu sigma] = featureNormalize(Xshuffle);
Xtrain = Xnorm(1:round(dm * split), :);
Xval = Xnorm((round(dm * split) + 1):end, :);

fprintf('Xtrain dimension: (%d, %d)\n', size(Xtrain));
fprintf('Xval dimension: (%d, %d)\n', size(Xval));

[m, n] = size(Xtrain);
[mval, nval] = size(Xval);

X = [ones(m, 1) Xtrain];
y = ytrain;

% Add Bias vector which is a column of ones
Xval = [ones(mval, 1) Xval];


initial_theta = zeros(n + 1, 1);

cost_history = ones(1)

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
options = optimset('GradObj', 'on', 'MaxIter', 400, 'OutputFcn', @displayCost);
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Make predictions using optimal theta values
% rounding gives the result in either 0's or 1's since the ouput should be
% either 0 or 1
p = round(sigmoid(Xval * theta));

fprintf('Training Accuracy: %.2f%%\n', mean(double(p == yval)) * 100);

% This dataset is highly skewed so Accuracy alone is not a good metric, we need
% to calculate additional Performance metrics such as Precision, Recall and F1 Score
true_positive = sum(yval == 1 & p == 1);
false_positive = sum(yval == 0 & p == 1);
false_negative = sum(yval == 1 & p == 0);

precision = true_positive / (true_positive + false_positive);
recall = true_positive / (true_positive + false_negative);

F1 = 2 * precision * recall / (precision + recall);

fprintf('Precision: %f, Recall: %f, F1 Score: %f\n', precision, recall, F1);

