function [feature, lab] = xin_extra(im, win_size) 

[n,m,c] = size(im);
lab = im_meanshift(im, win_size);
[seg_num, none] = size(unique(lab));

feature = zeros(seg_num,c);
for i = 1:c
    for j = 1 : seg_num
        idx = double(lab == j);
        feature(j,i) = sum(sum(double(im(:,:,i)).*idx))./sum(sum(idx));
    end
   
end

end