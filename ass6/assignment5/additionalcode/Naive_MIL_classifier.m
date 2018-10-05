%% Naive MIL classifier implementation
% Imread dataset
imDir1 = 'sival_apple_banana/apple';
imDir2 = 'banana';
cd('..');
cd(imDir1);
apple = dir;
apple_object = {};
banana_object = {};
for i = 1:length(apple)
    if apple(i).isdir == 0
        apple_object = [apple_object, imread(apple(i).name)];
    end
end

cd('..');
cd(imDir2);
banana = dir;

for j = 1: length(banana)
    if banana(i).isdir == 0
        banana_object = [banana_object, imread(banana(i).name)];
    end
end
cd('..');
cd('..');
% extractinstances
apple_width = 40;
banana_width = 40;
[apple_lab, apple_bags] = extractinstances(apple_object,apple_width);
[banana_lab, banana_bags] = extractinstances(banana_object,banana_width);
% creates MIL dataset
bags = [apple_bags,banana_bags];

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

set = gendatmilsival(bags, [1*ones(60,1);2*ones(60,1)]);
bagid = getident(set,'milbag');

% training Fisher classifier and get labels
w = fisherc(set);
labels = labeld(set,w);
% majority vote
single_label = combineinstlabels(labels,bagid,length(bags));
%% Accuracy
error_a = 0;
error_b = 0;
for i = 1:length(single_label)/2
    if single_label(i) == 2
        error_a = error_a + 1;
    end
    
    if single_label(60+i) == 1
        error_b = error_b + 1;
    end
        
end
error_rate = (error_a + error_b)/(length(apple_object)+length(banana_object));
accuracy = 1 - error_rate;

%% Scatterplot of instances
scatterd(set,'gridded')
%hold on
%legend('apple', 'banana')
%title('Scatter plot of instances from the two classes')





