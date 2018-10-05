function [w,b] = SVM_SGD(x,y,C)
   
   [n,d]=size(x);
   if(n ~= size(y,1))
       disp('size is not correspondent');
   end
   
   epoch=1000;
   
   eta_1=1;
   eta_0=1;
   w=zeros(d,1);
   b=sum(y-x*w)/n; %compute cost function
   losses=[];
   for i=1:epoch
       eta=eta_0/(eta_1+epoch);
       tem=[x,y];
       tem=tem(randperm(size(tem,1)),:); 
       x=tem(:,1:d);  %get rampted data
       y=tem(:,d+1);    %get label
       
       loss=[];
       
       for k=1:n
           yk=y(k,1);   
           xk=x(k,:);   %extract the kth training point
           c=yk*(xk*w+b); % identify the correctnes of classfication  
           if(c<1)
               w=w+eta*yk*xk';    %update
               b=b+eta*yk;
               loss=[loss;1-c];
           end 
       end
       
       obj=w'*w/(2*n)+C*sum(loss);
       losses=[losses;obj];
       
   end
   
   disp(['objective value is ',num2str(obj)]);
   
   fig=figure(1);
   plot([1:epoch],losses,'r');
   legend('Loss after each epoch');
   saveas(fig,'Loss_SVM_SGD_100.png');
   end
