function single_label = combineinstlabels(labels,bagid,num_bags)
% labels            is the labels of instances
% bagid             is the id of each instances
% num_bags          is the number N of bags in total

% single_label      is the label of bags, N*1 vector
id = 1;
i = 1;
n = 1;
inst_labels = [];
single_label = zeros(num_bags,1);
% loop until sort out all the labels of instances into corresponding bags
while i <= length(bagid)
    
    % sort labels of instances into cooresponding bags
    if bagid(i) == id
        inst_labels = [inst_labels; labels(i)];
        
    % if bagid changes, do majority vote for the previous bag and create a new bag 
    else
        single_label(n) = mode(inst_labels);
        n = n + 1;
        id = id + 1;
        inst_labels = labels(i);
    end
    i = i + 1;
end
single_label(n) = mode(inst_labels);