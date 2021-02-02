clc;
clear all;
close all;

covar = [];
mean = [];
post_vect = [];
accuracy = 0;
total_size = 200;
data_set = 'data.mat';

%------------------------------------------------------------------------------------------------------------

%Bayes Multiclass
[accuracy,testing_data] = bayes_MC(data_set,total_size,accuracy,post_vect);
disp('Accuracy:');
disp((accuracy/size(testing_data,2))*100);

%------------------------------------------------------------------------------------------------------------


function training_data = get_subject_train(filename,total_size)
    images = load(filename);
    faces = images.face;
    data = [];
    for n = 1:total_size
        % Change 3*n as per the train set requirement. 
        % 3*n - illumnination , 3*n-1 - Expression , 3*n-2 - Neutral
        imageN = faces(:,:,3*n-2);
        imageE = faces(:,:,3*n-1);
        data = [data imageN(:) imageE(:)];
    end
    training_data = data;
end

function testing_data = get_subject_test(filename,total_size)
    images = load(filename);
    faces = images.face;
    data = [];
    for n = 1:total_size
        % Change 3*n as per the test set requirement. 
        % 3*n - illumnination , 3*n-1 - Expression , 3*n-2 - Neutral
        imageI = faces(:,:,3*n);
        data = [data imageI(:)];
    end
    testing_data = data;
end

function [accuracy,testing_data] = bayes_MC(data_set,total_size,accuracy,post_vect)


training_data = get_subject_train(data_set,total_size);
testing_data = get_subject_test(data_set,total_size);

for n = 1:total_size    
    for m = 1:total_size
        class = [training_data(:,2*m-1) training_data(:,2*m)];
        mean = sum(class,2)/size(class,2);
        covar = cov(class');
        noise = 0.7*eye(length(class),length(class));
        covar = covar+noise;
        inv_covar = pinv(covar);
        posterior = (1/sqrt(2*pi*det(covar)))*exp(-0.5*(testing_data(:,n)-mean)'*inv_covar*(testing_data(:,n)-mean));
        post_vect = [post_vect;posterior];
    end
    [~,index] = max(post_vect);
    if index == n
        accuracy = accuracy + 1;
        fprintf('%d. correct\n',n);
    else
        fprintf('%d. incorrect\n',n);
    end
    post_vect = [];
end
end