function [lab, bag_rgb] = extractinstances(images,width)

N = length(images); % number of images
lab = {}; 
% get label for each image
for i = 1:N
   lab{i} = im_meanshift(images{i},width);
end

%% average RGB
bag_rgb = cell(60,1);

for i = 1:60
    % get each image
    image = images{i};
    % get the number of instances in current image
    num_ins = max(max(lab{i})); 
    % initialize N*3 matrix for each bag
    ins = zeros(num_ins,3);
    
    for n = 1:num_ins
        
        for c = 1:3
        ins_r = sum(sum(image(:,:,1).*double(lab{i}==n)))/sum(sum(lab{i}==n));
        ins_g = sum(sum(image(:,:,2).*double(lab{i}==n)))/sum(sum(lab{i}==n));
        ins_b = sum(sum(image(:,:,3).*double(lab{i}==n)))/sum(sum(lab{i}==n));
        end
        
        ins(n,:) = [ins_r, ins_g, ins_b];
    end
    % store the N*3 matrix in a cell set
    bag_rgb{i} = ins; 
end
end
