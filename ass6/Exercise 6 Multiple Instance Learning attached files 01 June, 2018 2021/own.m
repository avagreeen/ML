a_img=cell(60,1);
for i=1:length(apple_object)
    a_img{i}=rgb2hsv(apple_object{i})*256;
    
end

b_img=cell(60,1);
for i=1:length(apple_object)
    b_img{i}=rgb2hsv(banana_object{i})*256;
    
end
bags=a_img;
label=[ones(60,1);2*ones(60,1)];
%%
%[data,bags]=gendatmilsival(a_img, b_img, 30);
[a_lab, a_bags] = extractinstances(a_img,30);
[b_lab, b_bags] = extractinstances(b_img,30);
%%
bags=cell(120,1);
for i=1:60
   bags{i} =  a_bags{i};
   bags{60+i} = b_bags{i};
end


set = gendatmilsival_l(bags, [1*ones(60,1);2*ones(60,1)]);
bagid = getident(set,'milbag');
%%
scatterd(set,3)
%%
set_dat=getdata(set);
set_lab=getlab(set);
idx=find(set_dat(:,1)<50);
new_data=set_dat(idx);
new_lab=set_lab(idx);
new=prdataset(new_data,new_lab);

W=nmc(new);
labels = labeld(new,W);
acc=sum((labels-new_lab)==0)/length(labels)


