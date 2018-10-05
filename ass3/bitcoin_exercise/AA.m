% Exercise: Aggregating Algorithm (AA)

clear all;
load coin_data;

d = 5;
n = 213;

% compute adversary movez z_t
z_t=-log(r);
% compute strategy p_t (see slides)
L = zeros(n,d); 
p_t=zeros(n,d);  

for i=1:n
    if(i==1)
        p=1/d*[1 1 1 1 1];
        L(i,:)= z_t(1:i,:);
    else
        L(i,:)=sum(z_t((1:i),:));
        p=exp(-L(i-1,:));
    end
    p_t(i,:)=p/sum(p);
    mix_loss(i,:)=-log(p_t(i,:)*r(i,:)');
end
% compute loss of strategy p_t 
loss=sum(mix_loss,2);
% compute losses of experts
e_loss=min(L,[],2);

regret=loss-e_loss;
% compute total gain of investing with strategy p_t
gain=1/exp(sum(loss))
%% plot of the strategy p and the coin data

% if you store the strategy in the matrix p (size n * d)
% this piece of code will visualize your strategy

figure
subplot(1,2,1);
plot(p_t)
legend(symbols_str)
title('rebalancing strategy AA')
xlabel('date')
ylabel('confidence p_t in the experts')

subplot(1,2,2);
plot(s)
legend(symbols_str)
title('worth of coins')
xlabel('date')
ylabel('USD')
