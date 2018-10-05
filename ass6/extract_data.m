imDir1 = 'sival_apple_banana/apple';
imDir2 = 'banana';
cd(imDir1);
apple = dir;
apple_object = {};
banana_object = {};
for i = 3:length(apple)
    if apple(i).isdir == 0
        apple_object = [apple_object, imread(apple(i).name)];
    end
end

cd('..');
cd(imDir2);
banana = dir;

for j = 3: length(banana)
    if banana(i).isdir == 0
        banana_object = [banana_object, imread(banana(i).name)];
    end
end
cd('..');
cd('..');
%%
for i=1:length(apple_object)
    I = apple_object{1,i};
    img = imresize(I, 0.3);
    apple_object{1,i}=img;
end
%%
for i=1:length(banana_object)
    I = banana_object{1,i};
    img = imresize(I, 0.3);
    banana_object{1,i}=img;
end
