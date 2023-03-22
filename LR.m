% Implementing Logistic Regression model for classification

close all; clear all; clc;

%data = readmatrix('data.csv');
data = readmatrix('data_without_duplicates.csv');

%%
x = data(:, 2:10);
y = data(:, 11);

%%
% calling a function to split the data into train(80%) and test(20%)
data_split(x,y);
%%
load xtrain.mat;
load ytrain.mat;
load xtest.mat;
load ytest.mat;

B = mnrfit(xtrain,ytrain);

%%

figure;
%subplot(221);
stem(0:9,B);
xlabel('feature');
ylabel('weight');

%%
% validating on training set
pihat = mnrval(B,xtrain);

[~,yhat] = max(pihat,[],2);

table(pihat, yhat, ytrain);

% Accuracy
train_acc = mean(ytrain==yhat);

% ROC curve for training set
figure;
[fpr,tpr,~,AUC] = perfcurve(ytrain,pihat(:,2),2);
plot(fpr,tpr,'LineWidth',2);
xlabel('False positive rate');
ylabel('True positive rate');
title(['AUC on training set = ' num2str(AUC,'%0.4f')])


%%
% testing the model on actual test set

pihat_test = mnrval(B,xtest);

[~,yhat_test] = max(pihat_test,[],2);

table(pihat_test, yhat_test, ytest);

%%
% Accuracy
test_acc = mean(ytest==yhat_test);

% ROC curve for test set
figure;
[fpr,tpr,~,AUC] = perfcurve(ytest,pihat_test(:,2),2);
plot(fpr,tpr,'LineWidth',2);
xlabel('False positive rate');
ylabel('True positive rate');
title(['AUC on testing set = ' num2str(AUC,'%0.4f')])

%Confusion matrix

C = confusionmat(ytest, yhat_test);
figure;
cm = confusionchart(ytest,yhat_test);
tp = C(1,1);
fp = C(2,1);
tn = C(2,2);
fn = C(1,2);

accuracy = (tp + tn) / (tp + fp + tn + fn);
precision = tp / (tp + fp);  % precision
recall = tp / (tp + fn);  % sensitivity
spec = tn / (tn + fp);  % specificity

f1 = (2 * precision * recall) / (precision + recall);




