apple_width = 30;
banana_width = 30;
%%
imDir1 = 'sival_apple_banana/apple';
imDir2 = 'banana';
cd(imDir1);
apple = dir;
cd('..');
cd(imDir2);
banana = dir;

banana_object=cell(60,1);
for i = 3: length(banana)
    banana_object{i-2} = imresize(imread(banana(i).name),0.3);
end

apple_object=cell(60,1);
for i = 3: length(banana)
        apple_object{i-2} = imresize(imread(apple(i).name),0.3);
end
%% get pr data set
[data,bags]=gendatmilsival(apple_object, banana_object, 30)

%%
lab=getlab(data)
%%
bagid = getident(data,'milbag');
w = nmc(data);
labels = labeld(data,w);
%%
[test,train,idtst,idtrn]=gendat(data,0.2);

w = nmc(data);
error=testc(test,w);
pre=labeld(data,w);
%%
single_label = combineinstlabels(pre,bagid,length(data));
