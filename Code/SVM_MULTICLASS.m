% clc;
% clear all;
% close all;

data = load('illumination.mat');
total_size = 68;
data_set = data.illum;
train_size = 20;
theta = [];
theta0 = [];
train_data = [];
test_data = [];
accuracy = 0;
%------------------------------------------------------------------------------------------------------------

%SVM MULTICLASS  
[accuracy,test_data] = SVM_M(total_size,data_set,train_size,theta,theta0,train_data,test_data,accuracy);
disp('SVM MULTICLASS Acccuracy: ');
disp((accuracy/size(test_data,2))*100);

%------------------------------------------------------------------------------------------------------------

function [accuracy,test_data] = SVM_M(total_size,data_set,train_size,theta,theta0,train_data,test_data,accuracy)
for j = 1:total_size
    for i = 1:train_size 
        train_data = [train_data data_set(:,i,j)];
    end
    test_data = [test_data data_set(:,21,j)];
end

% Training

for j = 1:train_size:size(train_data,2)
    class = [];
    rem_data = train_data;
    for n = j:j+train_size-1 
        class = [class train_data(:,n)];
        rem_data(:,j) = [];
    end
    X = [class rem_data]';
    labels = [ones(size(class,2),1); -ones(size(rem_data,2),1)];
    H = (X*X').*(labels*labels');
    f = -ones(size(X,1),1);
    B = [labels';zeros(size(X,1)-1,size(X,1))];
    Beq = zeros(size(X,1),1);
    lb = zeros(size(X,1),1);
    ub = 1*ones(size(lb));
    mu = quadprog(H,f,[],[],B,Beq,lb,ub);
    theta = [theta ((mu.*labels)'*X)'];
    [~,index] = max(mu);
    theta0 = [theta0 (1/labels(index))-theta(:,end)'*X(index,:)'];
end

% Testing
for i = 1:size(test_data,2) 
    vect = [];
    for j = 1:size(theta,2)
        value = theta(:,j)'*test_data(:,i)+theta0(j);
        if value > 0
            vect = [vect 1];
        else
            vect = [vect 0];
        end
    end
    
    if sum(vect) == 1
        [~,index] = max(vect);
        if index == i
            disp('correct');
            accuracy = accuracy+1;
        end
    else
        disp('incorrect');
    end
end
end


