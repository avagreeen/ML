function [label, bag_minmax] = extractinstancesminmax(images,width)
% images           a cell of multi-images   
% width            width-parameter for im_meanshift
% label            a cell for label of each image
% bag_minmax       a cell with N*num_ins*6 matrix as extracted instances

N = length(images); % number of images
label = {}; 

for i = 1:N
   label = [label, im_meanshift(images{1,i},width)];
end

%% min and max of RGB

bag_minmax = {};

for i = 1:N
    image = images{1,i};
    num_ins = max(max(label{i})); % number of instance
    instance = zeros(num_ins,6); % matrix for instance for each image
    
    for n = 1:num_ins        
        ins_rmin = min(min(image(:,:,1).*uint8(label{i}==n)));
        ins_rmax = max(max(image(:,:,1).*uint8(label{i}==n)));
        ins_gmin = min(min(image(:,:,2).*uint8(label{i}==n)));
        ins_gmax = max(max(image(:,:,2).*uint8(label{i}==n)));
        ins_bmin = min(min(image(:,:,3).*uint8(label{i}==n)));
        ins_bmax = max(max(image(:,:,3).*uint8(label{i}==n)));
        
        instance(n,:) = [ins_rmin, ins_rmax, ins_gmin, ins_gmax, ins_bmin, ins_bmax];
    end
    
    bag_minmax = [bag_minmax, instance]; 
end
end
