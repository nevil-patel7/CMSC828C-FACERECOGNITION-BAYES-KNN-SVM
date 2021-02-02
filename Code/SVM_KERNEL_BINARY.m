clc;
clear all;
close all;

% this is the total size of the subjects.
total_size = 200;
% this is the size of subjects again, 150 data points in each class.
training_size = 100;
data_set = 'data.mat';

% this simply partitions the .mat file into training_size and testing_size.
% but the samples are still interleaved.
training_data = get_training_data(data_set, training_size);
testing_data = get_testing_data(data_set, total_size, training_size);

class_neutral = [];
class_expression = [];
for n = 1: training_size
    %3*n-2 is how neutral faces are indexed.
    class_neutral = [class_neutral training_data(:,3*n-2)];
    %3*n-1 is how expression faces are indexed.
    class_expression = [class_expression training_data(:,3*n-1)];
end
size(class_neutral)
% partitioning testing_set as well so it is easier to determine
% accuracy
testing_set_N = [];
testing_set_E = [];
for n = 1: total_size-training_size
    % ignoring the illumination class, appending only the neutral and
    % expression class. The illumination class is ignored (discarded).
    testing_set_N = [testing_set_N testing_data(:,3*n-2)];
    testing_set_E = [testing_set_E testing_data(:,3*n-1)];
end
% joining the two test set so 1st half are neutral points and 2nd half are
% expression points.
testing_set = [testing_set_N testing_set_E];


% SVM using QuadProg starts here -->
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% concatenating both the classes in the following way
X = [class_neutral class_expression]';
disp('size of X');
disp(size(X));
%% 
pos_label = ones(1,size(class_neutral,2))';
neg_label = -ones(1,size(class_expression,2))';
Y = [pos_label;neg_label];
disp('Size of Y');
disp(size(Y));
%%
Kernel_Cell={'linear';'ploynomial';'RBF';'Sigmoid'};

%%
global poly_con gamma kappa1 kappa2 precision Cost
poly_con=1; % For Polynomial Kernel
gamma=10000;% For RBF
kappa1=1/size(X,1);kappa2=kappa1; % For Sigmoid

precision=10^-8;Cost=0.5;
%%
% Step 3: Fit the model
% Choose the kernel
kernel=char(Kernel_Cell(3));
%
[alpha,Ker,beta0]=SVM(X,Y,kernel);
disp('Size of theta');
disp(size(alpha));
%%
% the values of mu obtained are not absolutely zero, but they are not
% comparable to the bigger values as well. Hence after a careful
% observation, the threshold is set to 10e-8 and all the mu's below the
% threshold are reduced to zeros.
alpha_ = [];
for i = 1:size(alpha,1)
    if alpha(i) <= 10^-8
        alpha_(i) = 0;
    else
        alpha_(i) = alpha(i);
    end
end
% mu_ is an appropriate vector with small values reduced to zeros.
alpha_ = alpha_';

% obtaining the values of wt. vector and bias term for linear
% classification
theta = ((alpha_.*Y)'*X)';
%%

accuracy = SVMtesting(theta,beta0,testing_set);
disp('base accuracy:');
disp(accuracy);


function testing_data = get_testing_data(filename, total_size, training_size)
    images = load(filename);
    faces = images.face;
    data = [];
    for n = (training_size*3)+1:total_size*3
        image = faces(:,:,n);
        image = image(:);
        data = [data image];
    end
    testing_data = data;
end

function training_data = get_training_data(filename, size)
    images = load(filename);
    faces = images.face;
    data = [];
    for n = 1:size*3
        image = faces(:,:,n);
        image = image(:);
        data = [data image];
    end
    training_data = data;
end

function accuracy = SVMtesting(theta,theta0,testing_set)
    acc = 0;
    disp('In SVMTEST');
    disp('theta');
    disp(size(theta));
    disp('test');
    disp(size(testing_set));
    disp(size(testing_set));
    for i = 1:size(testing_set,2)
        % asssigning true labels: +1 for upto 1st half of the testing_set
        % and -1 for the 2nd half of the testing_set
        if i <= size(testing_set,2)/2
            true_label = 1;
        else
            true_label = -1;
        end
        % using the test image in the linear predictor
        value = theta'*testing_set(:,i) + theta0;
        % multiplying value with the true_label for easy comparison to
        % evaluate accuracy
        prediction = value*true_label;
        
        % self-explainatory
        if prediction > 0
            acc = acc+1;
        end
    end
    accuracy = (acc/size(testing_set,2))*100;
end

%June11,2016
%SVM
function [alpha,Ker,beta0]=SVM(X,Y,kernel)
% X is N*p, Y is N*1,{-1,1}
% Constant=Inf for Hard Margin
global  precision Cost

switch kernel
    case 'linear'
        Ker=Ker_Linear(X,X);
    case 'ploynomial'
        Ker=Ker_Polynomial(X,X);
    case 'RBF'
        Ker=Ker_RBF(X,X);
    case 'Sigmoid'
        Ker=Ker_Sigmoid(X,X);
end

N= size(X,1);
size(X)
H= diag(Y)*Ker*diag(Y);
disp('H');
disp(size(H));
f= - ones(N,1);
disp('f');
disp(size(f));
Aeq=[Y';zeros(size(X,1)-1,size(X,1))];
disp('Aeq');
disp(size(Aeq));
beq=zeros(size(X,1),1);
disp('Beq');
disp(size(beq));
A=[];
b=[];
lb = zeros(N,1);
ub = Cost*ones(size(lb));
size(lb)
size(ub)
alpha=quadprog(H,f,[],[],Aeq,beq,lb,ub);
disp('alpha');
disp(size(alpha));


serial_num=(1:size(X,1))';
serial_sv=serial_num(alpha>precision&alpha<Cost);
disp('ser_sv');
disp(serial_sv);

temp_beta0=0;
for i=1:size(serial_sv,1)
    temp_beta0=temp_beta0+Y(serial_sv(i));
    temp_beta0=temp_beta0-sum(alpha(serial_sv(i))*...
        Y(serial_sv(i))*Ker(serial_sv,serial_sv(i)));
end
beta0=temp_beta0/size(serial_sv,1);
end

function Y=Ker_Sigmoid(X1,X2)
global kappa1 kappa2
Y=zeros(size(X1,1),size(X2,1));%Gram Matrix
for i=1:size(X1,1)
    for j=1:size(X2,1)
        Y(i,j)=(kappa1*dot(X1(i,:),X2(j,:))+kappa2);
    end
end
end

function Y=Ker_RBF(X1,X2)
global gamma
Y=zeros(size(X1,1),size(X2,1));%Gram Matrix
for i=1:size(X1,1)
    for j=1:size(X2,1)
        Y(i,j)=exp(-(1/(gamma^2))*norm(X1(i,:)-X2(j,:))^2);
    end
end
end

function Y=Ker_Polynomial(X1,X2)
global poly_con
Y=zeros(size(X1,1),size(X2,1));%Gram Matrix
for i=1:size(X1,1)
    for j=1:size(X2,1)
        Y(i,j)=(1+dot(X1(i,:),X2(j,:))).^poly_con;
    end
end
end

function Y=Ker_Linear(X1,X2)
Y=zeros(size(X1,1),size(X2,1));%Gram Matrix
for i=1:size(X1,1)
    for j=1:size(X2,1)
        Y(i,j)=dot(X1(i,:),X2(j,:));
    end
end
end














