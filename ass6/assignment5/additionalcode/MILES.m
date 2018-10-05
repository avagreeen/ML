%% separate training and testing data
load('apple_bags.mat');
load('banana_bags.mat');
bags = [apple_bags, banana_bags];

n1 = randperm(60);
n2 = randperm(60)+ 60*ones(1,60);
train = {};
test = {};

for i = 1:30
    train{i} = bags{1,n1(i)};
    train{i+30} = bags{1,n2(i)};
end

for i = 1:30
    test{i} = bags{1,n1(i+30)};
    test{i+30} = bags{1,n2(i+30)};
end

instances = gendatmilsival(bags, [1*ones(60,1);2*ones(60,1)]);

% implement bagembed to get feature vectors
mB_train =  bagembed(train,instances.data);
mB_test =  bagembed(test,instances.data);

%% make prdataset
mB_train_dataset = prdataset(mB_train, [ones(30,1);2*ones(30,1)]);
mB_test_dataset = prdataset(mB_test,[ones(30,1);2*ones(30,1)]);

%% train Liknon classifier
W = liknonc(mB_train_dataset, 30);

%% test trained Liknon classifier
labels = labeld(mB_test_dataset,W);

%% accuracy
error_a = 0;
error_b = 0;

for i = 1:30
    if labels(i) == 2
        error_a = error_a + 1;
    end
    
    if labels(i+30) == 1
        error_b = error_b + 1;
    end
end

error_rate = (error_a+error_b)/60;
accuracy = 1-error_rate;
