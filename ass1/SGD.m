function [ rn, rp ] = SGD( x, y )
%SGD Summary of this function goes here
%   Detailed explanation goes here

    [n,d]=size(x);
   epoch=1000;
   eta_1=1;
   
   losses=[];
   for i=1:epoch
       
       eta=1/10000;
       
       tem=[x,y];
       tem=tem(randperm(size(tem,1)),:); 
       x=tem(:,1:d);  %get rampted data
       y=tem(:,d+1);    %get label
       
       loss=[];
       
       for k=1:n
           yk=y(k,1);   
           xk=x(k,:);   %extract the kth training point
           
           if(rn>rp)
                
           end
       
       
       end
       
   end
end

