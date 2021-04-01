format compact

% load data
load ad_data.mat
load feature_name.mat

% create bias vector, parameters, and the two result vectors
bias_vector = ones(size(X_test,1),1);
parameters = linspace(0,1,11);
parameters(1) = 1e-08;
results = zeros(1,length(parameters));
weights_count = zeros(1,length(parameters));

% loop to make prediction using L1 regularization regression with the
% selected parameters
for i = 1:length(parameters)
%    get the weights accorfing to the parameter
    [weights,bias] = logistic_l1_train(X_train,y_train,parameters(i));
%    make prediction
    prediction = X_test*weights + bias_vector*bias;
    
%    count the number of features selected
    counter = 0;
    for j = i:length(weights)
        if ~isequal(weights(j),0)
            counter = counter+1;
        end
    end
    weights_count(i) = counter;
    
%    parse the prediction to +1/-1
    for j = 1:length(prediction)
        if prediction(j)<0
            prediction(j) = -1;
        else
            prediction(j) = 1;
        end
    end
    
    [X,Y,T,AUC] = perfcurve(y_test,prediction,1);
    results(i) = AUC(1);
end

% plot left - parameters vs AUC
yyaxis left
plot(parameters,results)
xlabel('parameter')
ylabel('AUC')
title('l1 regularized logistic regression')

% plot right - parameters vs number of features
yyaxis right
plot(parameters,weights_count)
ylabel('number of features')


function [w, c] = logistic_l1_train(data, labels, par)
    opts.rFlag = 1; % range of par within [0, 1].
    opts.tol = 1e-6; % optimization precision
    opts.tFlag = 4; % termination options.
    opts.maxIter = 5000; % maximum iterations
    [w, c] = LogisticR(data, labels, par, opts);
end
