function e = adaboost_error(X, y, k, a, d, alpha)
% adaboost_error: returns the final error rate of a whole adaboost
% 
% Input
%     X     : n * p matrix, each row a sample
%     y     : n * 1 vector, each row a label
%     k     : iter * 1 vector,  selected dimension of features
%     a     : iter * 1 vector, selected threshold for feature-k
%     d     : iter * 1 vector, 1 or -1
%     alpha : iter * 1 vector, weights of the classifiers
%
% Output
%     e     : error rate      

%%% Your Code Here %%%
    [n,~]=size(X);
    k=k(k>0);
    a=a(a~=0);
    d=d(d~=0);
    alpha=alpha(alpha~=0);
    cha=X(:,k)-repmat(a',[n,1]);
    g=((cha<=0)-0.5)*2*(alpha.*d);
    g=((g>0)-0.5)*2;
    e=length(find(g~=y))/n;
%%% Your Code Here %%%

end