function D = distance_test(test,train)
n = length(test);
m = length(train);
D = zeros(n,m);

for i = 1:n
    Bi = test{i};
    num_Bi = size(Bi,1);
    
    for j = 1:m
        
        Bj = train{j};
        num_Bj = size(Bj,1);
        s = zeros(num_Bi,num_Bj);
        
        for t1 = 1:num_Bi
            for t2 = 1:num_Bj
                s(t1,t2) = norm(Bi(t1,:) - Bj(t2,:))^2;
            end
        end
        
        D(i,j) = min(min(s));
    end
end