[n,m,non]=size(apple_object{1})
for i=1:60
    img=apple_object{i};
    for j=1:n
        for k=1:m
            I = double(reshape(img(j,k,:),3,1))
        end
    end
    
end