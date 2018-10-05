function [distance] = bagdistance(bag1, bag2)
% bag1, bag2   two image bag with multi-instance
% distance     the distance between two bags

[num_ins1, ~] = size(bag1);
[num_ins2, ~] = size(bag2);

distance_multi = zeros(num_ins1, num_ins2);

% calculate distance between each pair of instances
for i = 1:num_ins1
    for j = 1:num_ins2
        distance_multi(i,j) = norm(bag1(i,:)-bag2(j,:));
    end
end
% min distance
distance = min(min(distance_multi));
end