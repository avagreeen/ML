imDir1 = 'sival_apple_banana/apple';
imDir2 = 'banana';

cd(imDir1);
x = dir;
apple = [];
apple_img = {};
% read apple images
for i = 1:length(x)
    if x(i).isdir == 0
        apple = [apple; x(i)];
        apple_img = [apple_img, imread(x(i).name)];
    end
end

cd('..');
cd(imDir2);
y = dir;
banana = [];
banana_img = {};
% read banana images
for j = 1: length(y)
    if y(i).isdir == 0
        banana = [banana; y(i)];
        banana_img = [banana_img, imread(y(i).name)];
    end
end







