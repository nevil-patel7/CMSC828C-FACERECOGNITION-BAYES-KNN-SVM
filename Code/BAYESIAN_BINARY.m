clc;
clear all;
close all;

c_neutral = [];
c_expression = [];
test_n = [];
test_e = [];
accuracy = 0;
s_size = 200;
training_size = 100;
data_set = 'data.mat';

%------------------------------------------------------------------------------------------------------------

%Bayes 
[accuracy,test_set,c_neutral,c_expression] = bayes(data_set,training_size,s_size,c_neutral,c_expression,test_n,test_e,accuracy);
disp('Bayes Acccuracy: ');
disp((accuracy/size(test_set,2))*100);


%Bayes + MDA 
theta = MDA('data.mat');
accuracy = MDA_bayesian(theta,c_neutral,c_expression,test_set);
disp('Bayes + MDA acccuracy: ');
disp((accuracy/size(test_set,2))*100);


%Bayes + PCA 
comp = PCA([c_neutral c_expression],71);
accuracy = PCA_bayesian(comp,c_neutral,c_expression,test_set);
disp('Bayes + PCA acccuracy: ');
disp((accuracy/size(test_set,2))*100);

%------------------------------------------------------------------------------------------------------------

function training_data = f_train_data(filename, size)
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

function testing_data = f_test_data(filename, s_size, training_size)
    images = load(filename);
    faces = images.face;
    data = [];
    for n = (training_size*3)+1:s_size*3
        image = faces(:,:,n);
        image = image(:);
        data = [data image];
    end
    testing_data = data;
end

function accuracy = MDA_bayesian(theta,c1,c2,test_set)
    classNeutr = c1;
    classExprss = c2;
    training_size = size(c1,2);
    
    proj_classNeutr = theta' * classNeutr;
    proj_classExprss = theta' * classExprss;

    mean_Neutr = sum(proj_classNeutr)/training_size;
    mean_Exprss = sum(proj_classExprss)/training_size;

    cov_neutr = cov(proj_classNeutr');
    cov_exprss = cov(proj_classExprss');

    inv_cov_neutr = inv(cov_neutr);
    inv_cov_exprss = inv(cov_exprss);

    det_cov_neutr = det(cov_neutr);
    det_cov_exprss = det(cov_exprss);

    acc = 0;
    for n = 1:size(test_set,2)
        if n <= size(test_set,2)/2
            true_label = 1;
        else
            true_label = -1;
        end
        
        P_Neutr = (1/sqrt(2*pi*det_cov_neutr))*exp(-0.5*(theta'*test_set(:,n)-mean_Neutr)'*inv_cov_neutr*(theta'*test_set(:,n)-mean_Neutr));
        P_Exprss = (1/sqrt(2*pi*det_cov_exprss))*exp(-0.5*(theta'*test_set(:,n)-mean_Exprss)'*inv_cov_exprss*(theta'*test_set(:,n)-mean_Exprss));

        post = [P_Neutr 1;P_Exprss -1];
        [~,index] = max(post(:,1));

        if index == 1
            computed_label = 1;
        elseif index == 2
            computed_label = -1;
        end

        if true_label*computed_label == 1
            acc = acc+1;
        end
    end
    accuracy = acc;
end

function accuracy = PCA_bayesian(comp,c1,c2,test_set)
    classN = comp'*c1;
    classE = comp'*c2;
    
    meanN = sum(classN,2)/size(classN,2);
    meanE = sum(classE,2)/size(classE,2);
    
    covN = cov(classN');
    covE = cov(classE');
    
    invCovN = pinv(covN);
    invCovE = pinv(covE);
    
    I = eye(size(covN));
    noise = 0.001*I;
    covN = covN + noise;
    covE = covE + noise;
    
    detN = det(covN);
    detE = det(covE);
    
    acc = 0;
    for n = 1:size(test_set,2)
        if n <= size(test_set,2)/2
            true_label = 1;
        else
            true_label = -1;
        end
        
        P_Neutr = (1/sqrt(2*pi*detN))*exp(-0.5*((comp'*test_set(:,n)-meanN))'*invCovN*(comp'*test_set(:,n)-meanN));
        P_Exprss = (1/sqrt(2*pi*detE))*exp(-0.5*((comp'*test_set(:,n)-meanE))'*invCovE*(comp'*test_set(:,n)-meanE));

        post = [P_Neutr 1;P_Exprss -1];
        [~,index] = max(post(:,1));

        if index == 1
            computed_label = 1;
        elseif index == 2
            computed_label = -1;
        end
        
        if true_label*computed_label == 1
            acc = acc+1;
        end
    end
    accuracy = acc;
end

function theta = MDA(data_set)
    s_size = 200;
    training_size = 150;

    training_data = f_train_data(data_set, training_size);
    testing_data = f_test_data(data_set, s_size, training_size);

    c_neutral = [];
    c_expression = [];
    for n = 1: training_size
        c_neutral = [c_neutral training_data(:,3*n-2)];
        c_expression = [c_expression training_data(:,3*n-1)];
    end

    %MDA begins:
    %calculating covariance of c1 and c2 (neutral and expression)
    cov_neutral = cov(c_neutral');
    cov_expression = cov(c_expression');
    cov_matrix = cov_neutral + cov_expression;
    inv_cov_matrix = pinv(cov_matrix);
    mean_neutral = sum(c_neutral, 2)/training_size;
    mean_expression = sum(c_expression, 2)/training_size;
    final_mean = mean_neutral - mean_expression;
    theta = inv_cov_matrix * final_mean; 
end

function out = PCA(training_data,number)
    coeff = pca(training_data');
    coeff = coeff(:,1:number);
    out = coeff;
end

function [accuracy,test_set,c_neutral,c_expression] = bayes(data_set,training_size,s_size,c_neutral,c_expression,test_n,test_e,accuracy)

training_data = f_train_data(data_set, training_size);
testing_data = f_test_data(data_set, s_size, training_size);

for n = 1: training_size
    c_neutral = [c_neutral training_data(:,3*n-2)];
    c_expression = [c_expression training_data(:,3*n-1)];
end

for n = 1: s_size-training_size
    %Appending only the neutral and expression class. 
    %The illumination class is ignored.
    test_n = [test_n testing_data(:,3*n-2)];
    test_e = [test_e testing_data(:,3*n-1)];
end

test_set = [test_n test_e ];

mean_neutral = sum(c_neutral,2)/size(c_neutral,2);
mean_expression = sum(c_expression,2)/size(c_expression,2);

cov_neutral = cov(c_neutral');
cov_expression = cov(c_expression');

I = eye(size(cov_neutral));
noise = 0.4*I;
cov_neutral = cov_neutral + noise;
cov_expression = cov_expression + noise;

inv_cov_neutral = pinv(cov_neutral);
inv_cov_expression = pinv(cov_expression);

for n = 1:size(test_set,2)
    if n <= size(test_set,2)/2
        true_label = 1;
    else
        true_label = -1;
    end
    P_neutral = (1/sqrt(2*pi*det(cov_neutral)))*exp(-0.5*(test_set(:,n)-mean_neutral)'*inv_cov_neutral*(test_set(:,n)-mean_neutral));
    P_expression = (1/sqrt(2*pi*det(cov_expression)))*exp(-0.5*(test_set(:,n)-mean_expression)'*inv_cov_expression*(test_set(:,n)-mean_expression));
    
    posteriors = [P_neutral 1;P_expression -1];
    [~,index] = max(posteriors(:,1));
    
    if index == 1
        computed_label = 1;
    elseif index == 2
        computed_label = -1;
    end
    
    if true_label*computed_label == 1
        accuracy = accuracy+1;
    end
end
end


