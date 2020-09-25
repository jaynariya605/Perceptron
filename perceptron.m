clear all;clc; % clear output

%============================== Variable Declaration============================================
[~, data] = halfmoon(10,6,0,3000); % taking data from halfmoon.m provided where halfmoon(rad =10,width=6,d=0,n_samp=1000+2000)
n = etaseries(0.9,1E-5,1000); % it will give list of learning rate where etaseries( startpoint, end point, No of rates)
bias = 5; % bias
weight    = [bias;zeros(2,1)];% initialize weights

%=============================Training Perceptron using generated data=============================
for epoch = 1:50, % No of Epoch is 50
    shuffle_training_data = data(:,randperm(1000)); % Shuffle data and take 1000 samples from 3000 as training
    miss = 0;
    for i = 1:1000, % for one epoch for no of instance
        X_train = [1 ; shuffle_training_data(1:2,i)]; % getting input X training data from dataset 
        d = shuffle_training_data(3,i);         % getting true label from dataset
        Y_train = sigmoid(weight'*X_train); % find predicted value Y = sig(W.X)
        error(i) = d-Y_train; % find error e = ( d - Y)
        
        weight_update = weight + n(i)*(d-Y_train)*X_train; % Calculate update weight using W(n+1)= W(n) + eta.( d - Y). X
        weight = weight_update; % make update weight as weight W(n) = W(n+1)
        if (Y_train - shuffle_training_data(3,i)) ~= 0, % calculate error rate for training
            miss = miss + 1;
        end
    end
    
    mse(epoch) = mean(error.^2); % calculate Mean Square Error per epoch
    Accuracy = ((1000-miss)/1000)*100; % calculate Training Accuracy
    fprintf(' For epoch %f Training Accuracy is %f \n',epoch, Accuracy);
    fprintf(' For epoch %f Training Error is %f \n',epoch, 100- Accuracy);
end

%======================Ploating Data Points For Training Samples=========================

f = figure('visible','off');
hold on;
for i=1:1000,
    if shuffle_training_data(3,i) == 1,
        plot(shuffle_training_data(1,i),shuffle_training_data(2,i),'r+');
    else,
        plot(shuffle_training_data(1,i),shuffle_training_data(2,i),'kx');
    end
end



%==================Testing Dataset Using Trained Perceptron=====================
miss = 0;

for i = 1 : 2000, % for 2000 samples testing perceptron
    X_test = [1 ; data(1:2,i+1000)]; % getting input X testing data from dataset
    Y_test(i) = sigmoid(weight'*X_test); % find predicted value Y = sig(W.X) 
    if Y_test(i) == 1 , % plot data if predicted label is 1
        plot(data(1,i+1000),data(2,i+1000),'r+');
    end
    if Y_test(i) == -1, % plot data if predicted label is -1
        plot(data(1,i+1000),data(2,i+1000),'kx');
    end
     if (Y_test(i) - data(3,i+1000)) ~=0,% Find error using ( Y pred - Y true)
        miss = miss + 1;
     end
end

    

    
fprintf(' No of Missclassified Points is : %d \n', miss);
Accuracy = ((2000-miss)/2000)*100; % Testing Accuracy 
fprintf(' Testing Accuracy is %f \n', Accuracy);
fprintf(' Testing Error is %f \n', 100-Accuracy);

%=======================Ploting Halfmoon with Bounndary====================
xmin = min(data(1,:));
xmax = max(data(1,:));
ymin = min(data(2,:));
ymax = max(data(2,:));
xlim([xmin-1 xmax+1]);
ylim([ymin-1 ymax+1]);
yline(0,'k-');
set(f, 'visible', 'on'); 

%===========================================================================