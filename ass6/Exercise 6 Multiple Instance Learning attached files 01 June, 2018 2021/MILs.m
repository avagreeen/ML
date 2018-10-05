n1 = randperm(60);  %前60
n2 = randperm(60)+ 60*ones(1,60); %后60
train = cell(60,1);
test = cell(60,1);


for i = 1:30
    train{i} = bags{n1(i)};
    train{i+30} = bags{n2(i)};
end

for i = 1:30
    test{i} = bags{n1(i+30)};
    test{i+30} = bags{n2(i+30)};
end
% implement bagembed to get feature vectors
mB_train =  bagembed(train,set);
mB_test =  bagembed(test,set);
all=bagembed(bags,set);
%% make prdataset
mB_train_dataset = prdataset(mB_train, [ones(30,1);2*ones(30,1)]);
mB_test_dataset = prdataset(mB_test,[ones(30,1);2*ones(30,1)]);
pr_all=prdataset(all,[ones(60,1);2*ones(60,1)]);
%%
W = liknonc(pr_all, 30);

labels = labeld(pr_all,W);

%single_label = combineinstlab=ls(labels,bagid,length(mB_test_dataset));
%%
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
accuracy = 1-error_rate