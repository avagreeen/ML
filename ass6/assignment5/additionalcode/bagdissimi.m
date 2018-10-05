function [bagdissimi] = bagdissimi(bags)
% bags                 a cell with multi-bags
% bagdissimi           bag dis-similarities

N = length(bags);

bagdissimi = zeros(N, N);

for i = 1:N
    for j = 1:N
        bagdissimi(i,j) = bagdistance(bags{i}, bags{j});
    end
end

end