clear all;clc;% clear output

%============================== Data Preprocessing============================================

Data = readtable('sonar.txt');% https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)
Data = Data(randperm(size(Data,1)),:); % Shuffle dataset
Data = table2cell(Data);
for i=1:208, % Give ALphabate value to integer
    Data(i,61) = cellfun(@double,Data(i,61),'uni',0);
end
Data = cell2table(Data);
for i=1:208, % Give Replace Label R with 1 and M with -1
    if table2array(Data(i, 61)) == 82,
        Data{i,61} = -1;
    else
        Data{i,61} = 1;
    end
end
Data = Data{:,:};

%============================== Variable Declaration============================================

n = etaseries(0.9,1E-5,83);  % it will give list of learning rate where etaseries( startpoint, end point, No of rates)
weight    = [1;zeros(60,1)];
test_data = Data(1:125,1:61); % Taking 60% data as testing
test_data =test_data';
tran_data = Data(126:208,1:61); % Taking 40% data as trainging
tran_data = tran_data';
Data = Data';

%=============================Training Perceptron using generated data=============================
for epoch = 1:50, % No of Epoch is 50
    miss = 0;
    for i = 1:83, % for one epoch for no of instance 83
        X_train = [1;tran_data(1:60,i)]; % getting input X training data from dataset 
        d = tran_data(61,i); % getting true label from dataset
        Y_train = sigmoid(weight'*X_train); % find predicted value Y = sig(W.X)
        error(i) = d-Y_train; % find error e = ( d - Y)
         
        weight_update = weight + n(i)*(d-Y_train)*X_train;  % Calculate update weight using W(n+1)= W(n) + eta.( d - Y). X
        weight = weight_update; % make update weight as weight W(n) = W(n+1)
        
        if (Y_train - tran_data(61,i)) ~= 0, % calculate error rate for training
        miss = miss + 1;
    end
        
    end
    mse(epoch) = mean(error.^2);
    fprintf(' For epoch %f mse is %d \n ',epoch,mse(epoch));
    Accuracy = ((125-miss)/125)*100; % calculate Mean Square Error per epoch
    fprintf(' For epoch %f Training Accuracy is %f \n',epoch, Accuracy);  % calculate Training Accuracy
    fprintf(' For epoch %f Training Error is %f \n',epoch, 100- Accuracy);
end

%==================Testing Dataset Using Trained Perceptron=====================

miss=0;
for i = 1 : 125, % for 125 samples testing perceptron
    X_test = [1;test_data(1:60,i)]; % getting input X testing data from dataset
    Y_test(i) = sigmoid(weight'*X_test); % find predicted value Y = sig(W.X) 
    if (Y_test(i) - test_data(61,i)) ~= 0, % Find error using ( Y pred - Y true)
        miss = miss + 1;
    end
  
end

Accuracy = ((125-miss)/125)*100;
fprintf(' Testing Accuracy is %f \n', Accuracy); % Testing Accuracy 
fprintf(' Testing Error is %f \n', 100-Accuracy);