clc;
clear all;
close all;

K = input('Enter the value of K: ');

%------------------------------------------------------------------------------------------------------------

%KNN MULTICLASS POSE  
[accuracy,test_data] = KNN_P(K);
disp('KNN POSE MULTICLASS Acccuracy: ');
disp((accuracy/size(test_data,2))*100);

%------------------------------------------------------------------------------------------------------------

%------------------------------------------------------------------------------------------------------------

%KNN MULTICLASS ILLUMINATION  
[accuracy,test_data] = KNN_I(K);
disp('KNN ILLUMINATION MULTICLASS Acccuracy: ');
disp((accuracy/size(test_data,2))*100);

%------------------------------------------------------------------------------------------------------------

function [accuracy,test_data] = KNN_P(K)
all_data = [];
all_labels = [];
accuracy = 0;
data = load('pose.mat');
total_size = 68;
train_size = 13;
for j = 1:total_size
    for i = 1:train_size
        img = data.pose(:,:,i,j);
        all_data = [all_data img(:)];
    end
    label = j*ones(1,train_size);
    all_labels = [all_labels label];
end

for n = 1:size(all_data,2)-1 % for all the testing_data
    dist_vect = [];
    test_data = all_data;
    test_labels = all_labels;
    test_data(:,n) = [];
    test_labels(:,n) = [];
    for m = 1:size(test_data,2) % for all the training_data
        dist = norm(test_data(:,m)-all_data(:,n));
        dist_vect = [dist_vect dist];
    end
    dist_vect = [dist_vect; test_labels];

    subj_number = [];
    subj_index = [];
    vote_vect = zeros(1,total_size);
    for k = 1:K
        [~,index] = min(dist_vect(1,:));
        subj_index = dist_vect(2,index);
        vote_vect(subj_index) = vote_vect(subj_index) + 1;
        subj_number = [subj_number dist_vect(2,index)];
        dist_vect(:,index) = [];
    end
    
    [~,index] = max(vote_vect);
    if index == all_labels(n)
        accuracy = accuracy + 1;
    end
end
end

function [accuracy,test_data] = KNN_I(K)
all_data = [];
all_labels = [];
accuracy = 0;
data = load('illumination.mat');
total_size = 68;
train_size = 21;
for j = 1:total_size
    for i = 1:train_size
        img = data.illum(:,i,j);
        all_data = [all_data img(:)];
    end
    label = j*ones(1,train_size);
    all_labels = [all_labels label];
end

for n = 1:size(all_data,2)-1 % for all the testing_data
    dist_vect = [];
    test_data = all_data;
    test_labels = all_labels;
    test_data(:,n) = [];
    test_labels(:,n) = [];
    for m = 1:size(test_data,2) % for all the training_data
        dist = norm(test_data(:,m)-all_data(:,n));
        dist_vect = [dist_vect dist];
    end
    dist_vect = [dist_vect; test_labels];

    subj_number = [];
    subj_index = [];
    vote_vect = zeros(1,total_size);
    for k = 1:K
        [~,index] = min(dist_vect(1,:));
        subj_index = dist_vect(2,index);
        vote_vect(subj_index) = vote_vect(subj_index) + 1;
        subj_number = [subj_number dist_vect(2,index)];
        dist_vect(:,index) = [];
    end
    
    [~,index] = max(vote_vect);
    if index == all_labels(n)
        accuracy = accuracy + 1;
    end
end
end
