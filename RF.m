% Implementing Random Forest model for classification

close all; clear all; clc;


load xtrain.mat;
load ytrain.mat;
load xtest.mat;
load ytest.mat;

%%
rng('default');
t = templateTree('MaxNumSplits',9);
Mdl1 = fitcensemble(xtrain, ytrain,'Method','Bag','Learners',t);

%% 
% validating on training set

[y_val, score] = predict(Mdl1,xtrain);
table(y_val,ytrain);

% Accuracy
train_acc = mean(ytrain==y_val);

% ROC curve for training set
figure;
[fpr,tpr,~,AUC] = perfcurve(ytrain,score(:,2),2);
plot(fpr,tpr,'LineWidth',2);
xlabel('False positive rate');
ylabel('True positive rate');
title(['AUC on training set = ' num2str(AUC,'%0.4f')])
%%
% testing the model on actual test set

[y_pred, score] = predict(Mdl1,xtest);
table(y_pred,ytest);

% Accuracy
test_acc = mean(ytest==y_pred);

% ROC curve for training set
figure;
[fpr,tpr,~,AUC] = perfcurve(ytest,score(:,2),2);
plot(fpr,tpr,'LineWidth',2);
xlabel('False positive rate');
ylabel('True positive rate');
title(['AUC on testing set = ' num2str(AUC,'%0.4f')])


%Confusion matrix

C = confusionmat(ytest, y_pred);
figure;
cm = confusionchart(ytest,y_pred);
tp = C(1,1);
fp = C(2,1);
tn = C(2,2);
fn = C(1,2);

accuracy = (tp + tn) / (tp + fp + tn + fn);
precision = tp / (tp + fp);  % precision
recall = tp / (tp + fn);  % sensitivity
spec = tn / (tn + fp);  % specificity

f1 = (2 * precision * recall) / (precision + recall);
