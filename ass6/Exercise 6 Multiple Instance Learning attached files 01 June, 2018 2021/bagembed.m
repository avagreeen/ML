function [ mB ] = bagembed( bags ,instances)
%BAGEMBED Summary of this function goes here
%   Detailed explanation goes here
sigma = 30;
n = length(bags);                     % number of bags
m = size(instances,1);                % number of all instances
mB = zeros(n,m);                      % initializa the feature vector
instances=getdata(instances);
for i = 1:n
    bag = bags{i,1};
    num_ins = size(bag,1);
  
    for j = 1:m 
        s = zeros(1,num_ins);
        
        for t = 1:num_ins
            xij = bag(t,:);
            %xij=xij{1};
            s(i) = exp(-(norm(xij - instances(j,:)))^2/(sigma^2));
        end
        mB(i,j) = max(s);
    end
end
