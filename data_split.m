function data_split(x,y)

    len = length(y);
    P = 0.8;  % percentage

    rng(13) % to get the same shuffle every time
    idx = randperm(len);  % generating random numbers for index

    % selecting 80% of the data to train
    xtrain = x( idx(1:round(P*len)), :);  
    ytrain = y( idx(1:round(P*len)), :);  

    % selecting the remaining 20% for test
    xtest = x( idx(round(P*len)+1:end), :);  
    ytest = y( idx(round(P*len)+1:end), :); 

    % converting the output class values from 2 & 4 to 1 & 2

    ytrain = (ytrain==4);
    ytrain = ytrain + 1;

    ytest = (ytest==4);
    ytest = ytest + 1;

    % saving the training and testing matices to be used from other modules
    save('xtrain.mat', "xtrain");
    save('ytrain.mat', "ytrain");
    save('xtest.mat', "xtest");
    save('ytest.mat', "ytest");

end