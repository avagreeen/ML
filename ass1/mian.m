clear all
close all


[data]=load('optdigitsubset.txt');

label0=repmat(-1,[554,1]);
label1=repmat(1,[571,1]);
labels=[label0; label1];

%%
load('optdigitsubset.txt');
[n,m]=size(optdigitsubset);
subset1 = (optdigitsubset(1:554,:))';
subset2 = (optdigitsubset(555:1125,:))';
lamda = 0;
cvx_begin
    variable A(m)
    variable B(m)
    minimize( 1/554*(sum(sum_square(subset1 - repmat(A,1,554)))) + 1/571*(sum(sum_square(subset2 - repmat(B,1,571)))) + lamda * norm(A - B, 1))
cvx_end