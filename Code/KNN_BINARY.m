clc;
clear all;
close all;

c_neutral = [];
c_expression = [];
test_n = [];
test_e = [];
accuracy = 0;
K = 6;
s_size = 200;
training_size = 150;
data_set = 'data.mat';

%------------------------------------------------------------------------------------------------------------

%KNN  
[accuracy,test_set,c_neutral,c_expression] = KNN_(K,data_set,training_size,s_size,c_neutral,c_expression,test_n,test_e,accuracy);
disp('KNN Acccuracy: ');
disp((accuracy/size(test_set,2))*100);

%KNN + MDA 
theta = MDA('data.mat');
accuracy = MDA_KNN(theta,K,c_neutral,c_expression,test_set);
disp('KNN + MDA acccuracy: ');
disp((accuracy/size(test_set,2))*100);

%KNN + PCA
comp = PCA([c_neutral c_expression],25);
accuracy = PCA_KNN(comp,K,c_neutral,c_expression,test_set);
disp('KNN + PCA acccuracy: ');
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

function computed_label = label(distance_vector, K)
F = distance_vector(:,1);
    [B,I] = sort(F,1);
    X = [];
    for m = 1:K
    X = [X distance_vector(I(m),2)];
    end
    computed_label = mode(X);
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

function accuracy = MDA_KNN(theta,K,c1,c2,test_set)
    proj_classNeutr = theta' * c1;
    proj_classExprss = theta' * c2;

    acc = 0;
    for n = 1: size(test_set, 2)
        distance_vector = [];
        if n <= size(test_set, 2)/2
            true_label = 1;
        else
            true_label = -1;
        end
        
        for m = 1: size(proj_classNeutr, 2)
            distance = norm(theta'*test_set(:,n)-proj_classNeutr(:,m));
            distance_vector = [distance_vector;[distance 1]];
        end
       
        for m = 1: size(proj_classExprss, 2)
            distance = norm(theta'*test_set(:,n)-proj_classExprss(:,m));
            distance_vector = [distance_vector;[distance -1]];
        end
        
        computed_label = label(distance_vector, K);

        if true_label*computed_label == 1
            acc = acc + 1;
        end
    end
    accuracy = acc;
end

function out = PCA(training_data,number)
    coeff = pca(training_data');
    coeff = coeff(:,1:number);
    out = coeff;
end

function accuracy = PCA_KNN(comp,K,c1,c2,test_set)
    classN = comp'*c1;
    classE = comp'*c2;
    
    acc = 0;
    for n = 1: size(test_set, 2)
        distance_vector = [];
        if n <= size(test_set, 2)/2
            true_label = 1;
        else
            true_label = -1;
        end
        
        for m = 1: size(classN, 2)
            distance = norm(comp'*test_set(:,n)-classN(:,m));
            distance_vector = [distance_vector;[distance 1]];
        end
     
        for m = 1: size(classE, 2)
            distance = norm(comp'*test_set(:,n)-classE(:,m));
            distance_vector = [distance_vector;[distance -1]];
        end
        
        computed_label = label(distance_vector, K);

        if true_label*computed_label == 1
            acc = acc + 1;
        end
    end
    accuracy = acc;
end

function [accuracy,test_set,c_neutral,c_expression] = KNN_(K,data_set,training_size,s_size,c_neutral,c_expression,test_n,test_e,accuracy)
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

test_set = [test_n test_e];

for n = 1: size(test_set, 2)
    distance_vector = [];
    if n <= size(test_set, 2)/2
        true_label = 1;
    else
        true_label = -1;
    end

    for m = 1: size(c_neutral, 2)
        distance = norm(test_set(:,n)-c_neutral(:,m));
        distance_vector = [distance_vector;[distance 1]];
    end
    
    for m = 1: size(c_expression, 2)
        distance = norm(test_set(:,n)-c_expression(:,m));
        distance_vector = [distance_vector;[distance -1]];
    end
    
    computed_label = label(distance_vector, K);
    
    if true_label*computed_label == 1
        accuracy = accuracy + 1;
    end
end
end
