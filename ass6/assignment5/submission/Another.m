%% Another MIL using bag representation and dis-similarities
%% read all the apple and banana images
% imDir1 = 'sival_apple_banana/apple';
% imDir2 = 'banana';
% cd('..');
% cd(imDir1);
% apple = dir; 
% apple_object = {};
% banana_object = {};
% % read apple images
% for i = 1:length(apple)
%     if apple(i).isdir == 0
%         apple_object = [apple_object, imread(apple(i).name)];
%     end
% end
% 
% cd('..');
% cd(imDir2);
% banana = dir;
% % read banana images
% for j = 1: length(banana)
%     if banana(i).isdir == 0
%         banana_object = [banana_object, imread(banana(i).name)];
%     end
% end
% 
% cd('..');

%% Bag representation
% extract instances using extractinstancesminmax
% width = 40;
% [apple_label, apple_bags] = extractinstancesminmax(apple_object,width);
% [banana_label, banana_bags] = extractinstancesminmax(banana_object,width);

% to make it simple, the apple and banana bags are saved in mat files
load('apple_another.mat');
load('banana_another.mat');
bags = [apple_bags, banana_bags];

% seperate the whole dataset as training data and testing data
apple_rand = randperm(60);
banana_rand = randperm(60)+ 60*ones(1,60);

train_data = {};
test_data = {};
% 60 training data
for i = 1:30
    train_data{i} = bags{1, apple_rand(i)};
    train_data{i+30} = bags{1, banana_rand(i)};
end
% 60 testing data
for i = 1:30
    test_data{i} = bags{1, apple_rand(i+30)};
    test_data{i+30} = bags{1, banana_rand(i+30)};
end

%% Main implementation of bag dis-similarities
% calculate bag dissimilarities
bagdis_train = bagdissimi(train_data);
bagdis_test = bagdissimi(test_data);

% make prdataset
bagdis_train_dataset = prdataset(bagdis_train, [ones(30,1); 2*ones(30,1)]);
bagdis_test_dataset = prdataset(bagdis_test, [ones(30,1); 2*ones(30,1)]);

% train Liknon classifier
C = liknonc(bagdis_train_dataset, 30);

%% test and estimate error rate
% test
labels = labeld(bagdis_test_dataset, C);

% error rate
error_a = 0;
error_b = 0;

for i = 1:30
    if labels(i) == 2
        error_a = error_a + 1; % apple misclassification
    end
    
    if labels(i+30) == 1
        error_b = error_b + 1; % banana misclassification
    end
end

error_rate = (error_a+error_b)/60
accuracy = 1-error_rate

