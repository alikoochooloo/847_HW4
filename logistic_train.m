format compact

% loading the data
load data.txt
load labels.txt


% changing the labels from +1/0 to +1/-1
for i = 1:size(labels,1)
    if isequal(labels(i,1),0)
        labels(i,1) = -1;
    end
end

% dividing the data to testing and training
data = [data, ones([size(data,1),1])];
training_data = data(1:2000,:);
testing_data = data(2000:end,:);
training_label = labels(1:2000,:);
testing_label = labels(2000:end,:);

% creating the training indexes for each step
training_values = [200,500,800,1000,1500,2000];
prediction_acc = zeros(length(training_values),1);

% looping to train and test the accuracy of the model for each index in training values
for i = 1:length(training_values)
    weights = logistic_regression(training_data(1:training_values(i),:),training_label(1:training_values(i)));
    
    prediction = testing_data*weights;
    
%    parsing the results of the prediction to +1/-1
    for j = 1:length(prediction)
        if prediction(j)<0
            prediction(j) = -1;
        else
            prediction(j) = 1;
        end
    end
    prediction_acc(i) = accuracy(prediction, testing_label);
end

%plotting the results
plot(training_values,prediction_acc)
xlabel('number of training samples')
ylabel('accuracy')
title('logistic regression')


function [weights] = logistic_regression(data, labels, epsilon, maxiter)

    if ~exist('epsilon','var')
%        third parameter does not exist, so default it to 0.00001
        epsilon = 1e-05;
    end
    if ~exist('maxiter','var')
%        forth parameter does not exist, so default it to 1000
        maxiter = 1000;
    end
    weights = zeros(size(data,2),1);
%    we choose a step size
    rate = 1;
    
%    loop to calculate weights
    for i = 1:maxiter
        
%        loop to calculate gradient
        gradient = 0;
        for j = 1:size(labels,1)
            exponent = exp(-labels(j)*weights'*data(j,:)');
            gradient = gradient - exponent*labels(j)*data(j,:)'/(1+exponent);
        end
        gradient = gradient/size(labels,1);
        
%        update the weights
        old_weights = weights; 
        weights = weights-rate*gradient;
        
%        to check if the absolute difference between two consecutive itterations of weights has reach below the epsilon values
        if abs_dif(weights,old_weights,epsilon) 
            disp('we have reached convergence')
            break
        end
    end
end

% function for calculating the absolute difference between two vectors
function result = abs_dif(first, second, epsilon)
    total = abs(first-second);
    if mean(total)> epsilon
        result = 0;
    else
        result = 1;
    end
        
    
end

% function for calculating the accuracy of labels
function result = accuracy(prediction, labels)
    misses = 0;
    total = prediction -labels;
    for i = 1:length(total)
        if ~isequal(total(i),0)
            misses = misses+1;
        end
    end
    result = 1-misses/length(total);
end