function [ lab, bag_rgb ] = extractinstance( images,width )
%EXTRACTINSTANCE Summary of this function goes here
%   Detailed explanation goes here
N = length(images); % number of images
lab={};
bag_rgb={}
for i=1:N
    img=images{i};
    lab{i}=im_meanshift(img,width);
    num_ins=length(unique(lab{i}));
    ins=zeros(num_ins,3);
    
    for n= 1:num_ins
        mask = uint8(lab{i}==n);
        r=mean(mean(img(:,:,1).*mask));
        g=mean(mean(img(:,:,2).*mask));
        b=mean(mean(img(:,:,3).*mask));
        
        ins(n,:)=[r,g,b];
    end
    bag_rgb{i}=ins;


end

