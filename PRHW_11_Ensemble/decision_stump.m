function [k, a, d] = decision_stump(X, y, w)
% decision_stump returns a rule ...
% h(x) = d if x(k) <= a, -d otherwise,
%
% Input
%     X : n * p matrix, each row a sample
%     y : n * 1 vector, each row a label
%     w : n * 1 vector, each row a weight
%
% Output
%     k : the optimal dimension
%     a : the optimal threshold
%     d : the optimal d, 1 or -1

% total time complexity required to be O(p*n*logn) or less
%%% Your Code Here %%%
    [~,m]=size(X);
    % select feature coordinate and threshold
    e=[];
    for i=1:m
        st=min(X(:,i));
        en=max(X(:,i));
        inter=(en-st)/1000;
        a=st+inter;
        while(a<en)
            e=[e;decision_stump_error(X,y,i,a,1,w),i,a,1]; %#ok<*AGROW>
            e=[e;decision_stump_error(X,y,i,a,-1,w),i,a,-1];
            a=a+inter;
        end
        fprintf('%d feature\n',i);

    end
    % select args with best error rate
    [~,I]=sort(e(:,1));
    k=e(I(1),2);
    a=e(I(1),3);
    d=e(I(1),4);
%%% Your Code Here %%%

end